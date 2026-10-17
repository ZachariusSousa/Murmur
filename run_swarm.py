"""Run a local two-node coordination experiment with two Dream model copies."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from run_dream import (
    MODEL_ID,
    create_parser as create_baseline_parser,
    load_dream_model_and_tokenizer,
    reset_cuda_peak_memory,
)
from experiment import ScenarioResult, run_comparison


def dropout_probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def remask_fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def load_two_dream_nodes(
    torch_module: object,
    auto_model: object,
    auto_tokenizer: object,
    quantization_config_class: object,
) -> tuple[object, object, object]:
    primary, tokenizer = load_dream_model_and_tokenizer(
        torch_module, auto_model, auto_tokenizer, quantization_config_class
    )
    peer, _ = load_dream_model_and_tokenizer(
        torch_module,
        auto_model,
        auto_tokenizer,
        quantization_config_class,
        tokenizer=tokenizer,
    )
    return primary, peer, tokenizer


def create_parser() -> argparse.ArgumentParser:
    parser = create_baseline_parser()
    parser.description = (
        "Compare one Dream node with two separately loaded proposal nodes."
    )
    parser.add_argument(
        "--peer-dropout",
        type=dropout_probability,
        default=0.5,
        help="Probability that the optional node is absent on a step (default: 0.5).",
    )
    parser.add_argument(
        "--remask-fraction",
        type=remask_fraction,
        default=0.05,
        help=(
            "Maximum fraction of generated positions to remask once in the "
            "single-node revision run (default: 0.05; 0 disables it)."
        ),
    )
    parser.set_defaults(max_vram_gb=16.0)
    for action in parser._actions:
        if action.dest == "max_vram_gb":
            action.help = "Fail if PyTorch peak reserved VRAM exceeds this limit (default: 16 GiB for two model copies)."
    return parser


def run_experiment(args: argparse.Namespace) -> None:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "dependencies are missing; run 'uv sync' in PowerShell"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but PyTorch cannot see an NVIDIA GPU; "
            "check the driver and the CUDA-enabled PyTorch installation"
        )

    torch.cuda.empty_cache()
    reset_cuda_peak_memory(torch)
    print(f"Model: {MODEL_ID}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Loading two separate 4-bit NF4 model copies...")
    model, peer_model, tokenizer = load_two_dream_nodes(
        torch, AutoModel, AutoTokenizer, BitsAndBytesConfig
    )
    with torch.inference_mode():
        for event in run_comparison(args, model, tokenizer, peer_model=peer_model):
            if not isinstance(event, ScenarioResult):
                continue
            print(f"\n=== {event.scenario} ===")
            print(event.text)
            print("Statistics:")
            print(f"  elapsed: {event.elapsed_seconds:.2f} s")
            print(f"  active nodes per step: {list(event.active_nodes_per_step)}")
            print(f"  optional-node proposals committed: {event.peer_commits}")
            print(f"  optional-node failures ignored: {event.peer_failures}")
            print(f"  tokens remasked: {event.remasked_tokens}")
            eos_position = (
                str(event.first_eos_position)
                if event.first_eos_position is not None
                else "none"
            )
            print(f"  first EOS token position: {eos_position}")
            print(f"  output-token differences from baseline: {event.token_differences}")
            print(f"  PyTorch peak allocated VRAM: {event.peak_allocated_gb:.2f} GiB")
            print(f"  PyTorch peak reserved VRAM: {event.peak_reserved_gb:.2f} GiB")


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[argparse.Namespace], None] = run_experiment,
) -> int:
    args = create_parser().parse_args(argv)
    try:
        runner(args)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
