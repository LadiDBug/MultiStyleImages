"""Rendering e pacchetto portabile del modulo, senza modelli o browser personali."""

import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from urllib.parse import unquote

from src.evaluation.aggregate_rq1 import digest, write_json
from src.evaluation.review_gallery import gallery_config, render_page, upgrade


class ReviewGalleryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gallery = self.root / 'gallery'
        (self.gallery / 'grids').mkdir(parents=True)
        sources = self.root / 'sources'
        paths = ['run/samples/a.png/input_original.png']
        paths += [f'run/samples/a.png/images/gta5/{c}/a_seed42.png'
                  for c in ('none', 'canny', 'depth', 'segmentation')]
        hashes = {}
        for relative in paths:
            p = sources / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'unchanged-source')
            hashes[relative] = digest(p)
        self.grid = self.gallery / 'grids/gta5_a_seed42.png'
        self.grid.write_bytes(b'unchanged-grid')
        self.manifest = {'cases': [{'image_id': 'a.png', 'style': 'gta5', 'set': 'random', 'reason': 'random'}],
                         'source_file_sha256': hashes, 'source_root': str(sources), 'display_seed': 42,
                         'selection_id': 'test', 'analysis_manifest_sha256': 'test-analysis',
                         'artifacts_sha256': {'grids/gta5_a_seed42.png': digest(self.grid)}}
        write_json(self.gallery / 'visual_selection.json', self.manifest)
        (self.gallery / 'index.html').write_text('old-gallery', encoding='utf-8')
        (self.gallery / 'annotations.csv').write_text('notes\nuser note\n', encoding='utf-8')

    def test_local_and_portable_pages_have_same_identity(self):
        local = gallery_config(self.gallery, self.manifest)
        portable = gallery_config(self.gallery, self.manifest, portable=True)
        self.assertEqual(local['dataset_id'], portable['dataset_id'])
        self.assertTrue(all(p.startswith('sources/') for p in portable['items'][0]['sources'].values()))

    def test_update_preserves_csv_and_grids_and_archive_includes_sources(self):
        config = upgrade(self.gallery, share_zip=True)
        self.assertEqual((self.gallery / 'annotations.csv').read_text(), 'notes\nuser note\n')
        self.assertEqual(self.grid.read_bytes(), b'unchanged-grid')
        self.assertEqual((self.gallery / 'index_before_forms.html').read_text(), 'old-gallery')
        self.assertIn(config['dataset_id'], (self.gallery / 'index.html').read_text(encoding='utf-8'))
        with zipfile.ZipFile(self.gallery / 'galleria_rq1_da_condividere.zip') as archive:
            self.assertIn('index.html', archive.namelist())
            portable = gallery_config(self.gallery, self.manifest, portable=True)
            for path in portable['items'][0]['sources'].values():
                self.assertIn(unquote(path), archive.namelist())
            self.assertNotIn('annotations.csv', archive.namelist())

    def test_config_cannot_close_its_script_tag(self):
        config = gallery_config(self.gallery, self.manifest)
        config['items'][0]['reasons'] = ['</script><script>untrusted()</script>']
        page = render_page(config)
        self.assertNotIn('</script><script>untrusted()', page)
        self.assertIn('\\u003c/script\\u003e', page)


if __name__ == '__main__':
    unittest.main()
