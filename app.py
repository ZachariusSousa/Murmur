"""Small browser UI for observing local Dream diffusion experiments."""

from __future__ import annotations

from html import escape
import os
from threading import Lock
from types import SimpleNamespace
from typing import Callable, Iterator

from experiment import ExperimentProgress, ScenarioResult, run_comparison
from run_swarm import load_two_dream_nodes


EXPERIMENT_CONCURRENCY_LIMIT = 1


class ModelCache:
    """Load Dream only after the first launch and reuse it for later runs."""

    def __init__(self, loader: Callable[[], tuple[object, object, object]]):
        self.loader = loader
        self._value: tuple[object, object, object] | None = None
        self._lock = Lock()

    def get(self) -> tuple[object, object, object]:
        with self._lock:
            if self._value is None:
                self._value = self.loader()
            return self._value


def _load_dream() -> tuple[object, object, object]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("dependencies are missing; run uv sync in PowerShell") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; PyTorch cannot see an NVIDIA GPU")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return load_two_dream_nodes(
        torch, AutoModel, AutoTokenizer, BitsAndBytesConfig
    )


def _validated_args(prompt: str, max_new_tokens: int, steps: int,
                    temperature: float, top_p: float, seed: int,
                    max_vram_gb: float, peer_dropout: float,
                    remask_fraction: float = 0.05) -> SimpleNamespace:
    if not prompt.strip():
        raise ValueError("Enter a prompt before launching the experiment.")
    if max_new_tokens <= 0 or steps <= 0:
        raise ValueError("Token count and diffusion steps must be positive.")
    if temperature < 0:
        raise ValueError("Temperature must be zero or greater.")
    if not 0 < top_p <= 1:
        raise ValueError("Top-p must be greater than 0 and at most 1.")
    if max_vram_gb <= 0:
        raise ValueError("VRAM limit must be a positive number.")
    if not 0 <= peer_dropout <= 1:
        raise ValueError("Peer dropout must be between 0 and 1.")
    if not 0 <= remask_fraction <= 1:
        raise ValueError("Remask fraction must be between 0 and 1.")
    return SimpleNamespace(
        prompt=prompt.strip(), max_new_tokens=int(max_new_tokens), steps=int(steps),
        temperature=float(temperature), top_p=float(top_p), seed=int(seed),
        max_vram_gb=float(max_vram_gb), peer_dropout=float(peer_dropout),
        remask_fraction=float(remask_fraction),
    )


def _metric_table(results: dict[str, ScenarioResult]) -> str:
    if not results:
        return "_No completed scenarios yet._"
    rows = [
        "| Scenario | Time | Peer commits | Remasked | First EOS token | Token diff vs baseline | Peak reserved VRAM |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results.values():
        rows.append(
            f"| {result.scenario} | {result.elapsed_seconds:.2f}s | "
            f"{result.peer_commits} | {result.remasked_tokens} | "
            f"{result.first_eos_position if result.first_eos_position is not None else 'none'} | "
            f"{result.token_differences} | "
            f"{result.peak_reserved_gb:.2f} GiB |"
        )
        rows.append(f"\n**{result.scenario} output**\n\n{result.text}\n")
    rows.append("\nMeasurements describe influence and runtime, not text quality.")
    return "\n".join(rows)


def _token_html(commits: list[object], max_new_tokens: int,
                prompt_length: int = 0, tokenizer: object | None = None) -> str:
    cells = []
    by_position = {commit.position - prompt_length: commit for commit in commits}
    for pos in range(max_new_tokens):
        commit = by_position.get(pos)
        if commit is None:
            cells.append(f'<span class="token masked">{pos}: ·</span>')
        else:
            style = "peer" if commit.source == "peer" else "primary"
            token = (tokenizer.convert_ids_to_tokens(commit.token_id)
                     if tokenizer is not None else str(commit.token_id))
            cells.append(
                f'<span class="token {style}" title="token id {commit.token_id}">'
                f'{pos}: {escape(str(token))}</span>'
            )
    return '<div class="token-grid">' + "".join(cells) + "</div>"


def _plot_frame(rows: list[dict[str, object]], y_column: str):
    import pandas as pd
    return pd.DataFrame(rows, columns=["step", y_column, "scenario"])


def create_launch_handler(model_loader: Callable[[], tuple[object, object, object]]):
    import gradio as gr

    cache = ModelCache(model_loader)

    def launch(prompt: str, max_new_tokens: int, steps: int, temperature: float,
               top_p: float, seed: int, max_vram_gb: float, peer_dropout: float,
               remask_fraction: float = 0.05,
               ) -> Iterator[tuple[object, ...]]:
        commits: dict[str, list[object]] = {}
        masks: dict[str, list[dict[str, object]]] = {}
        nodes: dict[str, list[dict[str, object]]] = {}
        results: dict[str, ScenarioResult] = {}
        try:
            args = _validated_args(prompt, max_new_tokens, steps, temperature,
                                   top_p, seed, max_vram_gb, peer_dropout,
                                   remask_fraction)
            model, peer_model, tokenizer = cache.get()
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                return_tensors="pt", return_dict=True, add_generation_prompt=True,
            )
            prompt_length = templated.input_ids.shape[1]
            import torch
            inference = torch.inference_mode()
            with inference:
                for event in run_comparison(
                    args, model, tokenizer, peer_model=peer_model
                ):
                    if isinstance(event, ExperimentProgress):
                        trace = event.trace
                        remasked = set(trace.remasked_positions)
                        commits[event.scenario] = [
                            commit for commit in commits.get(event.scenario, [])
                            if commit.position not in remasked
                        ]
                        commits.setdefault(event.scenario, []).extend(trace.committed_tokens)
                        masks.setdefault(event.scenario, []).append(
                            {"step": trace.step + 1, "masks": trace.masks_remaining,
                             "scenario": event.scenario}
                        )
                        nodes.setdefault(event.scenario, []).append(
                            {"step": trace.step + 1, "active nodes": trace.active_nodes,
                             "scenario": event.scenario}
                        )
                        status = (f"{event.scenario} · step {trace.step + 1}/{steps} · "
                                  f"{trace.masks_remaining} masks remaining · "
                                  f"{trace.active_nodes} node(s) active · "
                                  f"{len(trace.remasked_positions)} token(s) remasked")
                        mask_plot_update = gr.skip()
                        node_plot_update = gr.skip()
                        summary_update = gr.skip()
                    elif isinstance(event, ScenarioResult):
                        results[event.scenario] = event
                        status = f"Completed: {event.scenario}"
                        mask_plot_update = _plot_frame(
                            [row for group in masks.values() for row in group], "masks"
                        )
                        node_plot_update = _plot_frame(
                            [row for group in nodes.values() for row in group],
                            "active nodes",
                        )
                        summary_update = _metric_table(results)
                    else:
                        continue
                    yield (
                        status,
                        _token_html(commits.get(event.scenario, []), max_new_tokens,
                                    prompt_length, tokenizer),
                        mask_plot_update,
                        node_plot_update,
                        summary_update,
                    )
        except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
            yield (f"Experiment not completed: {escape(str(exc))}",
                   gr.skip(), gr.skip(), gr.skip(), gr.skip())

    launch.model_cache = cache
    return launch


CSS = """
:root { --bg: #080b12; --panel: #101722; --line: #203345; --cyan: #50e3d2; --pink: #f45ca7; }
body, .gradio-container { background: var(--bg) !important; color: #dce7ed !important; }
.gradio-container { max-width: 1180px !important; }
.panel { border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
.token-grid { display: flex; flex-wrap: wrap; gap: 5px; min-height: 40px; }
.token { border: 1px solid var(--line); padding: 4px 6px; border-radius: 4px; font: 12px ui-monospace, monospace; }
.token.masked { color: #728494; } .token.primary { color: var(--cyan); border-color: var(--cyan); }
.token.peer { color: var(--pink); border-color: var(--pink); }
button.primary { background: #153a43 !important; border: 1px solid var(--cyan) !important; }
"""


def build_app(model_loader: Callable[[], tuple[object, object, object]] = _load_dream):
    import gradio as gr

    launch = create_launch_handler(model_loader)
    with gr.Blocks(title="Murmur · diffusion lab") as demo:
        gr.Markdown("# MURMUR / DIFFUSION LAB\nLocal denoising experiments · proposal influence is not a quality score")
        gr.Markdown("**Two-node proposals run concurrently on the same GPU using separate quantized model copies.** They share the current masked sequence; this tests proposal influence, not multi-PC scaling.")
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["panel"]):
                prompt = gr.Textbox(label="Prompt", lines=4, placeholder="What should Dream generate?")
                with gr.Row():
                    tokens = gr.Slider(1, 256, value=64, step=1, label="New tokens")
                    steps = gr.Slider(1, 256, value=64, step=1, label="Diffusion steps")
                with gr.Row():
                    temperature = gr.Slider(0, 2, value=0.2, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.01, 1, value=0.95, step=0.01, label="Top-p")
                with gr.Row():
                    seed = gr.Number(value=0, precision=0, label="Seed")
                    vram = gr.Number(value=16, label="VRAM limit (GiB)")
                dropout = gr.Slider(0, 1, value=0.5, step=0.05, label="Peer dropout")
                remask = gr.Slider(
                    0, 0.25, value=0.05, step=0.01,
                    label="Remask budget fraction (single-node run)",
                    info="Each selected token is remasked at most once.",
                )
                run = gr.Button("Run comparison", variant="primary")
            with gr.Column(scale=2, elem_classes=["panel"]):
                status = gr.Markdown("Ready. The model loads only when you launch the first run.")
                token_view = gr.HTML(value=_token_html([], 64), label="Committed token IDs")
        with gr.Row():
            mask_plot = gr.LinePlot(x="step", y="masks", color="scenario", title="Masks remaining by step", height=230)
            node_plot = gr.LinePlot(x="step", y="active nodes", color="scenario", title="Active contributors by step", height=230)
        summary = gr.Markdown("_No experiment results yet._")
        outputs = [status, token_view, mask_plot, node_plot, summary]
        run.click(
            launch,
            inputs=[prompt, tokens, steps, temperature, top_p, seed, vram, dropout, remask],
            outputs=outputs,
            concurrency_limit=EXPERIMENT_CONCURRENCY_LIMIT,
        )
    return demo


def main() -> None:
    demo = build_app()
    is_space = bool(os.getenv("SPACE_ID"))
    demo.queue(default_concurrency_limit=EXPERIMENT_CONCURRENCY_LIMIT).launch(
        server_name="0.0.0.0" if is_space else "127.0.0.1",
        share=False,
        css=CSS,
    )


if __name__ == "__main__":
    main()
