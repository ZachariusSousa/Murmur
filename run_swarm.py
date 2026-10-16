"""Run a local two-node coordination experiment with one loaded Dream model."""

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


def create_parser() -> argparse.ArgumentParser:
    parser = create_baseline_parser()
    parser.description = (
        "Compare one Dream node with two locally simulated proposal nodes."
    )
    parser.add_argument(
        "--peer-dropout",
        type=dropout_probability,
        default=0.5,
        help="Probability that the optional node is absent on a step (default: 0.5).",
    )
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
    print("Loading one shared 4-bit NF4 model...")
    model, tokenizer = load_dream_model_and_tokenizer(
        torch, AutoModel, AutoTokenizer, BitsAndBytesConfig
    )
    with torch.inference_mode():
        for event in run_comparison(args, model, tokenizer):
            if not isinstance(event, ScenarioResult):
                continue
            print(f"\n=== {event.scenario} ===")
            print(event.text)
            print("Statistics:")
            print(f"  elapsed: {event.elapsed_seconds:.2f} s")
            print(f"  active nodes per step: {list(event.active_nodes_per_step)}")
            print(f"  optional-node proposals committed: {event.peer_commits}")
            print(f"  optional-node failures ignored: {event.peer_failures}")
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
