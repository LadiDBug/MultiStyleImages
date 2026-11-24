"""Ricrea bundle e notebook RQ2 dai sorgenti locali, senza modelli o rete."""
import hashlib
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks'
FILES = [f'src/evaluation/{name}.py' for name in (
    'evaluate', 'records', 'content', 'style', 'dino', 'lora_eval')]
FILES += ['configs/evaluation.toml']
members = {name: (ROOT / name).read_bytes() for name in FILES}
members['src/__init__.py'] = b''
members['src/evaluation/__init__.py'] = b''
manifest = {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}
members['rq2_bundle_manifest.json'] = json.dumps(manifest, indent=2).encode()
bundle = OUT / 'rq2_evaluation_code.zip'
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, data in members.items():
        entry = zipfile.ZipInfo(name, date_time=(2026, 11, 24, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(entry, data)
BUNDLE_HASH = hashlib.sha256(bundle.read_bytes()).hexdigest()
MANIFEST_HASH = hashlib.sha256(members['rq2_bundle_manifest.json']).hexdigest()
cells = []


def cell(kind, source):
    item = {'cell_type': kind, 'metadata': {}, 'source': textwrap.dedent(source).strip() + '\n',
            'id': f'rq2-{len(cells):02d}'}
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell('markdown', '''
# RQ2 — Evaluation dei rank LoRA 4, 8, 16 e 32

Valuta le immagini **già generate** nel dataset `dianapamfile/lorarankresult`.
Metriche: CLIP-I, CLIP-T, DINO-I e mIoU SegFormer, stesso protocollo RQ1.
Non esegue training o generazione. Non richiede i pesi LoRA.

Importare questo notebook su Kaggle, allegare il dataset dei risultati e il bundle
`rq2_evaluation_code.zip` come secondo dataset. Abilitare GPU e Internet.
Il bundle è riconosciuto anche se Kaggle lo estrae automaticamente.

Eseguire le celle in ordine: leggere il riepilogo dell'audit prima della cella GPU.
L'audit cerca tutti i run batch nelle sottocartelle, anche run distinti per condizione.
Richiede la stessa matrice completa sui quattro rank: nessuna selezione automatica
delle sole foto comuni. In caso di incompatibilità si ferma e salva gli audit disponibili.

**Baseline senza LoRA assente**: `none` indica assenza di ControlNet, con LoRA attiva.
I confronti finali sono fra rank, separatamente per condizione. CLIP-T è una similarità,
non una percentuale di stile. Gli intervalli bootstrap descrivono variabilità fra foto,
non fra repliche di training; nessuna correzione per confronti multipli.

Preparato il 24 novembre 2026. Non ancora eseguito sull'ambiente reale Kaggle.
''')
cell('code', '''
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, os, platform, re, shutil, subprocess, sys, zipfile

DATASET = Path('/kaggle/input/datasets/dianapamfile/lorarankresult')
RANK_ROOTS = {
    '4': str(DATASET / 'Rank_4-20261124T105010Z-1-001'),
    '8': str(DATASET / 'Rank_8-20261124T105011Z-1-001'),
    '16': str(DATASET / 'Rank_16-20261124T105013Z-1-001'),
    '32': str(DATASET / 'Rank_32-20261124T105014Z-1-001'),
}
# Lasciare None per cercare il bundle nei dataset allegati.
# Se ci sono più copie, specificare lo ZIP o rq2_bundle_manifest.json desiderato.
BUNDLE_PATH = None
# Per riprendere nella stessa sessione, impostare qui la cartella WORK stampata.
# Dopo un reset Kaggle, occorre riallestire l'ambiente; non si importano ZIP precedenti.
RESUME_WORK = None

assert sys.version_info >= (3, 11), 'Richiesto Python >= 3.11'
assert Path('/kaggle/working').is_dir(), 'Eseguire su Kaggle.'
for rank, path in RANK_ROOTS.items():
    assert Path(path).is_dir(), f'Percorso rank {rank} assente: {path}'
if RESUME_WORK:
    WORK = Path(RESUME_WORK).resolve()
    assert WORK.is_relative_to(Path('/kaggle/working')) and WORK.is_dir()
else:
    WORK = Path('/kaggle/working') / ('rq2_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    WORK.mkdir(exist_ok=False)
CODE = WORK / 'code'
RESULTS = WORK / 'results'
RESULTS.mkdir(exist_ok=True)
print('WORK =', str(WORK))
print('GPU usata soltanto nella cella di evaluation.')
''')
cell('markdown', '''
## 1. Codice verificato e dipendenze

Installa solo le dipendenze dei valutatori e conserva PyTorch fornito da Kaggle.
Gli originali restano sotto `/kaggle/input`; tutti gli output sono sotto `WORK`.
Se pip richiede il riavvio, ripartire dalla prima cella con `RESUME_WORK` impostato.
''')
cell('code', '''
EXPECTED_BUNDLE_HASH = 'BUNDLE_HASH_PLACEHOLDER'
EXPECTED_MANIFEST_HASH = 'MANIFEST_HASH_PLACEHOLDER'
if BUNDLE_PATH is None:
    candidates = sorted(Path('/kaggle/input').rglob('rq2_evaluation_code.zip'))
    if not candidates:
        candidates = sorted(Path('/kaggle/input').rglob('rq2_bundle_manifest.json'))
    assert len(candidates) == 1, f'Allegare il bundle o impostare BUNDLE_PATH; trovati: {candidates}'
    bundle_source = candidates[0]
else:
    bundle_source = Path(BUNDLE_PATH)
if bundle_source.suffix == '.zip':
    assert hashlib.sha256(bundle_source.read_bytes()).hexdigest() == EXPECTED_BUNDLE_HASH, 'Bundle diverso.'
    with zipfile.ZipFile(bundle_source) as archive:
        for name in archive.namelist():
            assert (CODE / name).resolve().is_relative_to(CODE.resolve())
        archive.extractall(CODE)
else:
    assert bundle_source.name == 'rq2_bundle_manifest.json'
    assert hashlib.sha256(bundle_source.read_bytes()).hexdigest() == EXPECTED_MANIFEST_HASH
    files = json.loads(bundle_source.read_text())
    CODE.mkdir(exist_ok=True)
    for name in files:
        assert (CODE / name).resolve().is_relative_to(CODE.resolve())
        target = CODE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle_source.parent / name, target)
    shutil.copy2(bundle_source, CODE / bundle_source.name)
assert hashlib.sha256((CODE / 'rq2_bundle_manifest.json').read_bytes()).hexdigest() == EXPECTED_MANIFEST_HASH
for name, expected in json.loads((CODE / 'rq2_bundle_manifest.json').read_text()).items():
    assert hashlib.sha256((CODE / name).read_bytes()).hexdigest() == expected, name
os.chdir(CODE)
sys.path.insert(0, str(CODE))
subprocess.run([sys.executable, '-m', 'pip', 'install', 'transformers==4.57.6',
                'huggingface-hub>=0.34,<1', 'numpy>=1.26,<3',
                'Pillow>=10,<14', 'safetensors>=0.4,<1'], check=True)
'''.replace('BUNDLE_HASH_PLACEHOLDER', BUNDLE_HASH).replace('MANIFEST_HASH_PLACEHOLDER', MANIFEST_HASH))
cell('markdown', '''
## 2. Audit senza modelli

Controlla file, metadati, rank/alpha, hash dell'adattatore, scala LoRA, revisione base,
foto, seed, prompt, scheduler, ControlNet e geometria degli output. Registra le firme
di training ma non può attestare l'identità dei dataset di training dai soli output:
per questo serve l'audit dei training separato. Errori o duplicati bloccano la valutazione.
La prima esecuzione può richiedere tempo per leggere e calcolare gli hash delle immagini.
''')
cell('code', '''
from src.evaluation.lora_eval import audit_ranks, discover_runs, read_json, file_hash, verify_result, aggregate
PLAN = RESULTS / 'audit' / 'plan.json'
if PLAN.exists():
    plan = read_json(PLAN)
    assert all([run['path'] for run in plan['ranks'][rank]['runs']] ==
               [str(p) for p in discover_runs(root)]
               for rank, root in RANK_ROOTS.items()), 'Percorsi diversi dal piano salvato.'
    for item in plan['ranks'].values():
        for run in item['runs']:
            assert file_hash(Path(run['path']) / 'run_config.json') == run['manifest_sha256']
    print('Piano già presente: le immagini saranno ricontrollate prima delle metriche.')
else:
    plan = audit_ranks(RANK_ROOTS, RESULTS / 'audit')
print(json.dumps({k: plan[k] for k in ('n_inputs', 'outputs_per_rank', 'conditions', 'seeds', 'note')}, indent=2))
for rank, item in plan['ranks'].items():
    print('Rank', rank, ':', len(item['runs']), 'run,', len(item['rows']), 'output, scala', item['adapter']['scale'])
''')
cell('markdown', '''
## 3. Congelare i valutatori e registrare l'ambiente

Risolve le revisioni Hugging Face una volta per tutti i rank. Per mantenere anche le
stesse revisioni usate in RQ1, allegare il suo `evaluation_frozen.toml` e impostare
`RQ1_PROTOCOL` qui sotto. In assenza, il protocollo usa gli stessi modelli e testi ma
risolve le revisioni al momento di questa sessione RQ2. Nessuna inferenza in questa cella.
''')
cell('code', '''
import torch, tomllib
from huggingface_hub import HfApi
from importlib import metadata

RQ1_PROTOCOL = None  # Esempio: Path('/kaggle/input/.../evaluation_frozen.toml')
PROTOCOL = RESULTS / 'evaluation_frozen.toml'
assert torch.cuda.is_available(), 'Selezionare una GPU Kaggle.'
if not PROTOCOL.exists():
    text = Path(RQ1_PROTOCOL).read_text(encoding='utf-8') if RQ1_PROTOCOL else (CODE / 'configs/evaluation.toml').read_text(encoding='utf-8')
    parsed = tomllib.loads(text)
    for section in ('clip', 'dino', 'semantic'):
        item = parsed[section]
        revision = HfApi().model_info(item['model_id'], revision=item['revision']).sha
        assert re.fullmatch(r'[0-9a-f]{40}', revision)
        pattern = rf'(\[{section}\][^\[]*?revision\s*=\s*)"[^\"]+"'
        text, count = re.subn(pattern, lambda m: m.group(1) + json.dumps(revision), text, count=1)
        assert count == 1
    PROTOCOL.write_text(text, encoding='utf-8')
protocol = tomllib.loads(PROTOCOL.read_text(encoding='utf-8'))
from src.evaluation.evaluate import METRICS
assert set(protocol['evaluation']['metrics']) == set(METRICS)
for section in ('clip', 'dino', 'semantic'):
    assert re.fullmatch(r'[0-9a-f]{40}', protocol[section]['revision'])
environment = {
    'python': sys.version, 'platform': platform.platform(), 'gpu': torch.cuda.get_device_name(0),
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'packages': {p: metadata.version(p) for p in ('transformers', 'huggingface-hub', 'numpy', 'Pillow')},
    'bundle_sha256': EXPECTED_BUNDLE_HASH, 'plan_sha256': file_hash(PLAN),
    'protocol_sha256': file_hash(PROTOCOL), 'dataset': str(DATASET),
    'revisions': {s: protocol[s]['revision'] for s in ('clip', 'dino', 'semantic')},
}
from src.evaluation.evaluate import dump_json
dump_json(RESULTS / ('environment_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '.json'), environment)
print(json.dumps(environment, indent=2))
''')
cell('markdown', '''
## 4. Evaluation sulla GPU Kaggle

Eseguire dopo avere letto l'audit. Un processo separato per rank libera la memoria GPU
al termine. Log e CSV parziali vengono conservati. I rank completati vengono verificati
e saltati se si riesegue la cella nella stessa sessione. Se un rank fallisce, incrementare
solo il suo numero in `ATTEMPTS` prima di rieseguire; non cancellare i tentativi precedenti.
Non c'è resume dentro un rank. Non si ricaricano automaticamente risultati dopo reset Kaggle.
''')
cell('code', '''
ATTEMPTS = {'4': 1, '8': 1, '16': 1, '32': 1}  # Incrementare solo i tentativi falliti.
DESTINATIONS = {rank: str(RESULTS / f'rank{rank}_attempt{ATTEMPTS[rank]}') for rank in RANK_ROOTS}
for rank in ('4', '8', '16', '32'):
    destination = Path(DESTINATIONS[rank])
    if (destination / 'completed.json').exists():
        verify_result(PLAN, PROTOCOL, rank, destination)
        print('Rank', rank, 'già completo e verificato.', flush=True)
        continue
    assert not destination.exists(), f'Tentativo incompleto: incrementare ATTEMPTS["{rank}"] e ricalcolare DESTINATIONS.'
    command = [sys.executable, '-u', '-m', 'src.evaluation.lora_eval',
               '--plan', str(PLAN), '--protocol', str(PROTOCOL), '--rank', rank,
               '--output-dir', str(destination)]
    print('Avvio rank', rank, flush=True)
    with (RESULTS / f'rank{rank}_attempt{ATTEMPTS[rank]}.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            status = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    assert status == 0, f'Rank {rank} fallito: vedere log. Esportare anche il parziale con l’ultima cella.'
    verify_result(PLAN, PROTOCOL, rank, destination)
print('Tutti i rank completati e verificati.')
''')
cell('markdown', '''
## 5. Confronti appaiati fra i sei abbinamenti di rank

`paired_between_ranks.csv`: delta **rank alto − rank basso**, a parità di foto,
condizione e seed. Prima media dei seed per foto, poi media e bootstrap sulle foto.
Nessun confronto contro un modello senza LoRA. I CSV `paired_controlnet_vs_none.csv`
nei singoli rank riguardano invece le condizioni ControlNet nello stesso rank.
''')
cell('code', '''
AGGREGATED = RESULTS / ('aggregate_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
aggregate(PLAN, PROTOCOL, DESTINATIONS, AGGREGATED)
print('CSV finali:', AGGREGATED)
print((AGGREGATED / 'summary.json').read_text())
''')
cell('markdown', '''
## 6. Esportazione completa o parziale

Questa cella può essere eseguita anche dopo un errore nelle metriche. Esporta risultati,
audit, configurazioni, codice e log; non include immagini o pesi. Un archivio è marcato
COMPLETE soltanto se tutti i rank sono verificati e l'aggregazione corrente esiste.
Scaricare lo ZIP prima di terminare la sessione Kaggle e conservarlo anche se PARTIAL.
''')
cell('code', '''
complete = False
try:
    for rank, destination in DESTINATIONS.items():
        verify_result(PLAN, PROTOCOL, rank, destination)
    assert (AGGREGATED / 'summary.json').is_file()
    assert AGGREGATED.is_relative_to(RESULTS)
    complete = True
except Exception as error:
    print('Esportazione PARTIAL:', error)
label = 'COMPLETE' if complete else 'PARTIAL'
dump_json(RESULTS / 'export_status.json', {'status': label, 'utc': datetime.now(timezone.utc).isoformat()})
archive_path = WORK / f'rq2_evaluation_{label}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")}.zip'
with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
    for folder in (RESULTS, CODE):
        for path in sorted(folder.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.write(path, path.relative_to(WORK))
print('Scaricare:', archive_path)
from IPython.display import FileLink, display
display(FileLink(str(archive_path)))
''')
notebook = {'cells': cells, 'metadata': {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'},
}, 'nbformat': 4, 'nbformat_minor': 5}
for item in cells:
    if item['cell_type'] == 'code':
        compile(item['source'], '<notebook cell>', 'exec')
(OUT / 'RQ2_EVALUATION_RANKS_KAGGLE.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print('Notebook compilato; bundle SHA256:', BUNDLE_HASH)
