"""Audit e confronti RQ2 tra rank; nessuna generazione o training."""

import argparse
from collections import defaultdict
import csv
import itertools
import json
import math
from pathlib import Path
import statistics
import tomllib

from src.evaluation.evaluate import (
    METRICS, dump_json, evaluate_rows, file_hash, mean_and_interval, summarize,
)
from src.evaluation.records import KEY_FIELDS, audit_run, read_json, write_csv


RANKS = (4, 8, 16, 32)


def key(row):
    return tuple(str(row[field]) for field in KEY_FIELDS)


def discover_runs(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Cartella assente: {root}")
    runs = []
    for path in sorted(root.rglob('run_config.json')):
        manifest = read_json(path)
        if manifest.get('mode') == 'generation' and manifest.get('input_images'):
            runs.append(path.parent)
    if not runs:
        raise ValueError(f"Nessun run batch di generazione in {root}. Servono run_config.json e samples/.")
    return runs


def audit_ranks(roots, destination):
    """Richiede l'intera matrice identica: nessuna intersezione silenziosa."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    if {int(r) for r in roots} != set(RANKS):
        raise ValueError('Servono i quattro rank 4, 8, 16, 32.')
    plan = {'ranks': {}, 'note': 'Quattro LoRA; baseline senza LoRA assente.'}
    common_settings = common_scale = None
    common_keys = common_inputs = common_geometry = None
    adapter_hashes = set()
    for rank in RANKS:
        rows_by_key, inputs, geometry, runs = {}, {}, {}, []
        adapter = None
        for run in discover_runs(roots[str(rank)]):
            manifest, rows, report = audit_run(run)
            label = f'rank{rank}_run{len(runs) + 1}'
            write_csv(destination / f'{label}_audit.csv', rows)
            dump_json(destination / f'{label}_audit_summary.json', report)
            dump_json(destination / f'{label}_manifest.json', manifest)
            if report['invalid_outputs']:
                raise ValueError(f'{run}: {report["invalid_outputs"]} output non validi; consultare audit.')
            if set(manifest['experiment']['styles']) != {'pixel_art'}:
                raise ValueError(f'{run}: atteso soltanto pixel_art.')
            lora = manifest.get('lora', {})
            if lora.get('rank') != rank or lora.get('alpha') != rank:
                raise ValueError(f'{run}: rank/alpha assente o diverso da {rank}.')
            if not lora.get('sha256') or not lora.get('training_signature') or not lora.get('base_model_revision'):
                raise ValueError(f'{run}: mancano hash adattatore, firma training o revisione base.')
            scale = lora.get('scale')
            if not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0:
                raise ValueError(f'{run}: scala LoRA non valida.')
            if common_scale is None:
                common_scale = scale
            if scale != common_scale:
                raise ValueError('Scale LoRA diverse fra run/rank.')
            identity = {f: lora.get(f) for f in ('rank', 'alpha', 'sha256', 'training_signature', 'base_model_revision', 'scale')}
            if adapter is not None and adapter != identity:
                raise ValueError(f'Adattatori diversi nello stesso rank {rank}.')
            adapter = identity
            for row in rows:
                condition = row['condition']
                settings = {f: manifest[f] for f in ('model', 'generation', 'preprocessing_enabled')}
                settings['prompt'] = manifest['styles']['pixel_art']
                settings['scheduler'] = row['scheduler']
                settings['base_model_revision'] = lora['base_model_revision']
                if condition != 'none':
                    settings['controlnet_checkpoint'] = manifest['controlnets'][condition]
                    settings['controlnet_scale'] = manifest['controlnet']['conditioning_scale']
                    settings['preprocessing'] = manifest['preprocessing'][condition]
                if common_settings is None:
                    common_settings = {}
                if condition in common_settings and settings != common_settings[condition]:
                    raise ValueError(f'Impostazioni diverse: {run}, {condition}.')
                common_settings[condition] = settings
                k = key(row)
                if k in rows_by_key:
                    raise ValueError(f'Output duplicato nel rank {rank}: {k}. Selezionare un solo tentativo per output.')
                sample = Path(row['input_path']).parent
                records = [json.loads(line) for line in (sample / 'metadata.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
                record = next(r for r in records if key(r) == k)
                if record.get('lora') != lora:
                    raise ValueError(f'Metadati LoRA diversi dal manifest: {sample}.')
                image_hash = file_hash(Path(row['input_path']))
                if row['image_id'] in inputs and inputs[row['image_id']] != image_hash:
                    raise ValueError(f'Input diverso tra run: {row["image_id"]}.')
                inputs[row['image_id']] = image_hash
                geometry[k] = (row['output_width'], row['output_height'])
                row['input_sha256'] = image_hash
                row['output_sha256'] = file_hash(Path(row['output_path']))
                rows_by_key[k] = row
            runs.append({'path': str(run), 'manifest_sha256': file_hash(run / 'run_config.json')})
        if common_keys is None:
            common_keys, common_inputs, common_geometry = set(rows_by_key), inputs, geometry
        if set(rows_by_key) != common_keys or inputs != common_inputs or geometry != common_geometry:
            raise ValueError(f'Rank {rank}: foto, seed, condizioni, hash input o dimensioni non appaiati.')
        if adapter['sha256'] in adapter_hashes:
            raise ValueError('Stesso hash adattatore associato a rank diversi.')
        adapter_hashes.add(adapter['sha256'])
        plan['ranks'][str(rank)] = {'runs': runs, 'adapter': adapter, 'rows': [rows_by_key[k] for k in sorted(rows_by_key)]}
    plan['settings'] = common_settings
    plan['n_inputs'] = len(common_inputs)
    plan['outputs_per_rank'] = len(common_keys)
    plan['conditions'] = sorted({k[2] for k in common_keys})
    plan['seeds'] = sorted({int(k[3]) for k in common_keys})
    # Ogni foto deve avere l'intero prodotto condizioni x seed.
    if len(common_keys) != len(common_inputs) * len(plan['conditions']) * len(plan['seeds']):
        raise ValueError('Matrice foto/condizioni/seed incompleta, anche se comune ai rank.')
    dump_json(destination / 'plan.json', plan)
    return plan


def read_scores(path, expected):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    found = {key(r): r for r in rows}
    if len(found) != len(rows) or set(found) != {key(r) for r in expected}:
        raise ValueError(f'CSV incompleto o duplicato: {path}')
    for original in expected:
        row = found[key(original)]
        if row['status'] != 'success':
            raise ValueError(f'Metrica fallita: {path}, {key(row)}')
        for field in ('input_sha256', 'output_sha256'):
            if row[field] != original[field]:
                raise ValueError(f'Hash diverso dall\'audit: {field}, {path}')
        for metric in METRICS:
            row[metric] = float(row[metric])
            if not math.isfinite(row[metric]):
                raise ValueError(f'Valore non finito: {path}, {metric}')
    return rows


def verify_result(plan_path, protocol_path, rank, destination):
    info = read_json(Path(destination) / 'provenance.json')
    if info != {'rank': int(rank), 'plan_sha256': file_hash(Path(plan_path)),
                'protocol_sha256': file_hash(Path(protocol_path))}:
        raise ValueError(f'Provenienza diversa: {destination}')
    expected = read_json(Path(plan_path))['ranks'][str(rank)]['rows']
    return read_scores(Path(destination) / 'per_image_metrics.csv', expected)


def evaluate_rank(plan_path, protocol_path, rank, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    plan = read_json(Path(plan_path))
    rows = plan['ranks'][str(rank)]['rows']
    # Ricontrolla gli hash immediatamente prima delle metriche.
    for row in rows:
        for kind in ('input', 'output'):
            if file_hash(Path(row[f'{kind}_path'])) != row[f'{kind}_sha256']:
                raise ValueError(f'File cambiato dopo audit: {row[f"{kind}_path"]}')
    with Path(protocol_path).open('rb') as stream:
        protocol = tomllib.load(stream)
    dump_json(destination / 'provenance.json', {
        'rank': int(rank), 'plan_sha256': file_hash(Path(plan_path)),
        'protocol_sha256': file_hash(Path(protocol_path)),
    })
    evaluate_rows(rows, protocol, list(METRICS), 'cuda', destination)
    read_scores(destination / 'per_image_metrics.csv', read_json(Path(plan_path))['ranks'][str(rank)]['rows'])
    summary, comparisons, _ = summarize(rows, plan['conditions'], list(METRICS))
    write_csv(destination / 'summary_metrics.csv', summary)
    write_csv(destination / 'paired_controlnet_vs_none.csv', comparisons,
              ['category', 'style', 'condition', 'metric', 'n_outputs', 'n_inputs', 'mean', 'std_across_inputs', 'ci95_low', 'ci95_high'])
    dump_json(destination / 'completed.json', {'rank': rank, 'outputs': len(rows)})


def aggregate(plan_path, protocol_path, destinations, destination):
    all_rows, summaries, comparisons = [], [], []
    plan = read_json(Path(plan_path))
    scores = {}
    for rank in RANKS:
        rows = verify_result(plan_path, protocol_path, rank, destinations[str(rank)])
        scores[rank] = {key(row): row for row in rows}
        summary, _, _ = summarize(rows, plan['conditions'], list(METRICS))
        summaries.extend(dict(rank=rank, **row) for row in summary)
        all_rows.extend(dict(rank=rank, **row) for row in rows)
    for low, high in itertools.combinations(RANKS, 2):
        groups = defaultdict(lambda: defaultdict(list))
        for k, first in scores[low].items():
            second = scores[high][k]
            for category in ('__all__', first['category']):
                for metric in METRICS:
                    groups[(category, first['condition'], metric)][first['image_id']].append(second[metric] - first[metric])
        for (category, condition, metric), inputs in sorted(groups.items()):
            comparisons.append(dict(rank_low=low, rank_high=high, direction='high_minus_low',
                                    category=category, condition=condition, metric=metric,
                                    n_outputs=sum(map(len, inputs.values())),
                                    **mean_and_interval([statistics.mean(v) for v in inputs.values()])))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination / 'all_rank_metrics.csv', all_rows)
    write_csv(destination / 'summary_by_rank.csv', summaries)
    write_csv(destination / 'paired_between_ranks.csv', comparisons)
    dump_json(destination / 'summary.json', {
        'n_inputs': plan['n_inputs'], 'outputs_per_rank': plan['outputs_per_rank'],
        'conditions': plan['conditions'], 'seeds': plan['seeds'],
        'baseline_without_lora': False,
        'note': 'Delta rank alto meno basso, a parita di foto/condizione/seed. Media seed per foto, bootstrap per foto; CI descrittivi senza correzione confronti multipli. Nessuna replica di training.',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--rank', type=int, choices=RANKS, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    evaluate_rank(args.plan, args.protocol, args.rank, args.output_dir)
