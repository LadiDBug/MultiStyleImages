# MultiStyleImages

### Eseguire la baseline

```powershell
python src/run_baseline.py --input_dir src/test_images --output_dir results/baseline --seed 42 --steps 25
```

Per una prova sulla prima immagine della cartella, in ordine alfabetico:

```powershell
python src/run_baseline.py --input_dir src/test_images --output_dir results/test --num_samples 1 --seed 42 --steps 25
```

Per scegliere una foto specifica, inserirla da sola in una cartella e
passare quella cartella a `--input_dir`.

Parametri disponibili:

- `--num_samples`: numero massimo di immagini.
- `--seed`: seed della generazione, default 42.
- `--steps`: step di diffusione, default 25.
- `--guidance_scale`: guidance del prompt, default 7.5.
- `--control_scale`: intensità di ControlNet, default 1.0.
- `--control_end`: frazione della generazione fino alla quale applicare ControlNet, default 0.85.

### Eseguire solo la segmentazione

```powershell
python src/segmentation.py "src/test_images/paesaggio-1024x683.png"
```

Con preprocessing:

```powershell
python src/segmentation.py "src/test_images/paesaggio-1024x683.png" --preprocess
```

La mappa viene salvata accanto alla foto con suffisso `_seg_map.png`.

## Risultati

La baseline salva nella directory di output:

- immagini generate nelle cartelle `cyberpunk`, `gta_5` e `pixel_art`;
- mappe semantiche nella cartella `segmentation_maps`;
- parametri e tempi in `experiment_metadata.json`.

Usare cartelle separate per input e output. La directory di input deve
contenere solo fotografie da elaborare, perché anche le mappe PNG vengono
considerate immagini di input.
