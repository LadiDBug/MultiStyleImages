"""RQ2: controlli offline su immagini e metriche sintetiche, senza modelli."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.evaluation.evaluate import METRICS, dump_json, file_hash
from src.evaluation.lora_eval import RANKS, aggregate, audit_ranks, read_scores
from src.evaluation.records import write_csv


def fixture(root):
    roots = {}
    for rank in RANKS:
        rank_root = root / str(rank)
        roots[str(rank)] = str(rank_root)
        # Run separati per condizione, annidati come ZIP Google Drive.
        for condition in ('none', 'canny'):
            run = rank_root / 'nested' / condition
            sample = run / 'samples' / 'scene' / 'photo.jpg'
            sample.mkdir(parents=True)
            lora = dict(rank=rank, alpha=rank, scale=1.0, sha256=str(rank),
                        training_signature=f'train{rank}', base_model_revision='basecommit')
            manifest = dict(mode='generation', input_images=['scene/photo.jpg'],
                            preprocessing_enabled=False, model={'id': 'test'}, generation={'steps': 50},
                            experiment=dict(styles=['pixel_art'], conditions=[condition], seeds=[42, 123]),
                            styles={'pixel_art': dict(prompt='pixel art', negative_prompt='')},
                            controlnets={'canny': 'control'}, controlnet={'conditioning_scale': 0.8},
                            preprocessing={'canny': {'low_threshold': 100}}, lora=lora)
            dump_json(run / 'run_config.json', manifest)
            Image.new('RGB', (64, 48)).save(sample / 'input_original.png')
            rows = []
            for seed in (42, 123):
                output = sample / 'images' / 'pixel_art' / condition / f'photo_seed{seed}.png'
                output.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (64, 48), (rank, 0, 0)).save(output)
                rows.append(dict(image_id='scene/photo.jpg', style='pixel_art', condition=condition,
                                 seed=seed, status='success', model=manifest['model'],
                                 generation=manifest['generation'], preprocessing_enabled=False,
                                 input_size=[64, 48], scheduler='Scheduler', prompt='pixel art', negative_prompt='',
                                 controlnet_checkpoint='control' if condition == 'canny' else None,
                                 controlnet_conditioning_scale=0.8 if condition == 'canny' else None, lora=lora))
            (sample / 'metadata.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    return roots


class RQ2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.roots = fixture(self.root)

    def test_nested_split_runs_and_pairwise_deltas(self):
        plan = audit_ranks(self.roots, self.root / 'audit')
        self.assertEqual(plan['outputs_per_rank'], 4)
        protocol = self.root / 'protocol.toml'
        protocol.write_text('# synthetic')
        plan_path = self.root / 'audit/plan.json'
        destinations = {}
        for rank in RANKS:
            dest = self.root / f'scores{rank}'
            dest.mkdir()
            destinations[str(rank)] = str(dest)
            rows = copy.deepcopy(plan['ranks'][str(rank)]['rows'])
            for row in rows:
                row.update(status='success', **{m: rank / 100 for m in METRICS})
            write_csv(dest / 'per_image_metrics.csv', rows)
            dump_json(dest / 'provenance.json', dict(rank=rank, plan_sha256=file_hash(plan_path), protocol_sha256=file_hash(protocol)))
        aggregate(plan_path, protocol, destinations, self.root / 'aggregate')
        with (self.root / 'aggregate/paired_between_ranks.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6 * 2 * 4 * 2)
        row = next(r for r in rows if r['rank_low'] == '4' and r['rank_high'] == '32')
        self.assertAlmostEqual(float(row['mean']), 0.28)
        self.assertEqual(row['n_inputs'], '1')
        expected = plan['ranks']['4']['rows']
        path = Path(destinations['4']) / 'per_image_metrics.csv'
        scores = read_scores(path, expected)
        scores[0]['clip_i'] = float('nan')
        write_csv(path, scores)
        with self.assertRaisesRegex(ValueError, 'non finito'):
            read_scores(path, expected)
        write_csv(path, scores[1:])
        with self.assertRaisesRegex(ValueError, 'incompleto'):
            read_scores(path, expected)

    def test_rejects_changed_input(self):
        for path in Path(self.roots['8']).rglob('input_original.png'):
            Image.new('RGB', (64, 48), 'red').save(path)
        with self.assertRaisesRegex(ValueError, 'non appaiati'):
            audit_ranks(self.roots, self.root / 'audit')

    def test_rejects_duplicate_runs(self):
        import shutil
        shutil.copytree(Path(self.roots['4']) / 'nested/none', Path(self.roots['4']) / 'duplicate')
        with self.assertRaisesRegex(ValueError, 'duplicato'):
            audit_ranks(self.roots, self.root / 'audit')

    def test_rejects_unpaired_settings_and_lora_metadata(self):
        path = Path(self.roots['8']) / 'nested/canny/run_config.json'
        manifest = json.loads(path.read_text())
        manifest['preprocessing']['canny']['low_threshold'] = 50
        dump_json(path, manifest)
        with self.assertRaisesRegex(ValueError, 'Impostazioni diverse'):
            audit_ranks(self.roots, self.root / 'audit')

    def test_rejects_metadata_with_different_adapter(self):
        path = Path(self.roots['16']) / 'nested/none/samples/scene/photo.jpg/metadata.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['lora']['sha256'] = 'wrong-adapter'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Metadati LoRA diversi'):
            audit_ranks(self.roots, self.root / 'audit')

    def test_rejects_same_incomplete_matrix_across_all_ranks(self):
        for rank in RANKS:
            run = Path(self.roots[str(rank)]) / 'nested/canny'
            manifest = json.loads((run / 'run_config.json').read_text())
            manifest['experiment']['seeds'] = [42]
            dump_json(run / 'run_config.json', manifest)
            path = run / 'samples/scene/photo.jpg/metadata.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            path.write_text(json.dumps(rows[0]) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Matrice.*incompleta'):
            audit_ranks(self.roots, self.root / 'audit')


if __name__ == '__main__':
    unittest.main()
