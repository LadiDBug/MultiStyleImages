"""Interfaccia Gradio, avviabile dal notebook o con python -m demo.app."""

import argparse
import logging

import gradio as gr
import torch

from demo.backend import DemoRuntime

CSS = """
.gradio-container {max-width: 1280px !important;}
#hero {padding: 24px 0 16px;}
#hero h1 {font-size: clamp(28px, 4vw, 44px); letter-spacing: -1.5px;}
#hero p {max-width: 700px; font-size: 16px;}
#generate {margin-top: 16px;}
"""


def build_app(runtime):
    if not runtime.ready:
        raise ValueError("Inizializza la demo con runtime.load() prima di creare il sito.")
    ranks = list(runtime.lora_paths)

    def lora_controls(style, enabled):
        visible = style == "pixel_art"
        return gr.update(visible=visible), gr.update(visible=visible and enabled)

    def generate(image, condition, style, use_lora, rank, seed):
        try:
            return runtime.run(image, condition, style, use_lora, rank, seed)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        except torch.OutOfMemoryError as exc:
            logging.exception("Memoria GPU esaurita durante la generazione")
            raise gr.Error("Memoria GPU insufficiente per questa immagine e mappa. Prova una foto più piccola o la mappa Nessuna. La demo conserva le dimensioni originali della foto.") from exc
        except Exception as exc:
            logging.exception("Generazione demo fallita")
            raise gr.Error("Generazione fallita. Controlla il log nel notebook; verifica anche la memoria GPU.") from exc

    with gr.Blocks(title="MultiStyle Studio", theme=gr.themes.Soft(primary_hue="indigo"), css=CSS) as app:
        gr.Markdown("# MultiStyle Studio\nUna foto, tre modi di immaginarla. Scegli lo stile e guida la trasformazione con una mappa strutturale.", elem_id="hero")
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("### 01 · La tua immagine")
                image = gr.Image(label="Carica una foto", type="pil", sources=["upload"], height=300)
                gr.Markdown("La foto viene passata alla pipeline senza crop o resize preliminari.")
                gr.Markdown("### 02 · La trasformazione")
                condition = gr.Radio([("Nessuna", "none"), ("Canny", "canny"), ("Depth", "depth"), ("Segmentation", "segmentation")], value="none", label="Tipo di mappa")
                style = gr.Radio([("GTA5", "gta5"), ("Pixel art", "pixel_art"), ("Watercolor", "watercolor")], value="gta5", label="Stile")
                with gr.Column(visible=False) as lora_panel:
                    use_lora = gr.Checkbox(label="Usa LoRA Pixel art", value=False, interactive=bool(ranks))
                    rank = gr.Dropdown(choices=ranks, value=ranks[0] if ranks else None, label="Rank del checkpoint", visible=False)
                    if not ranks:
                        gr.Markdown("Nessun peso LoRA trovato. Aggiungi i checkpoint e reinizializza la demo nel notebook.")
                with gr.Accordion("Riproducibilità", open=False):
                    seed = gr.Number(value=42, precision=0, minimum=0, maximum=2**32 - 1, label="Seed")
                button = gr.Button("Genera immagine", variant="primary", elem_id="generate")
                gr.Markdown("I modelli necessari vengono caricati a ogni generazione e rilasciati al termine. Il primo utilizzo può richiedere il download dei pesi.")
            with gr.Column(scale=5):
                gr.Markdown("### 03 · Il risultato")
                output = gr.Image(label="Immagine generata", type="pil", format="png", interactive=False, height=490)
                with gr.Accordion("Mappa utilizzata e dettagli", open=False):
                    map_output = gr.Image(label="Mappa strutturale (vuota con Nessuna)", type="pil", interactive=False)
                    details = gr.JSON(label="Dettagli generazione")
        style.change(lora_controls, [style, use_lora], [lora_panel, rank], queue=False)
        use_lora.change(lora_controls, [style, use_lora], [lora_panel, rank], queue=False)
        button.click(generate, [image, condition, style, use_lora, rank, seed],
                     [output, map_output, details], concurrency_limit=1,
                     concurrency_id="gpu", api_name=False)
    return app.queue(max_size=8, default_concurrency_limit=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Crea un link pubblico temporaneo Gradio.")
    args = parser.parse_args()
    runtime = DemoRuntime().load()
    build_app(runtime).launch(share=args.share, inbrowser=True)
