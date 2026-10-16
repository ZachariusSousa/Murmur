"""Run a local two-node coordination experiment with one loaded Dream model."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence

from run_dream import (
    MODEL_ID,
    create_parser as create_baseline_parser,
    cuda_peak_memory_gb,
    decode_response,
    enforce_vram_limit,
    load_dream_model_and_tokenizer,
    prepare_prompt,
    reset_cuda_peak_memory,
    validate_generated_text,
)
from swarm_sampler import SwarmResult, swarm_generate


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


def _token_differences(result: SwarmResult, baseline: SwarmResult,
                       prompt_length: int) -> int:
    generated = result.sequences[:, prompt_length:]
    baseline_generated = baseline.sequences[:, prompt_length:]
    return int((generated != baseline_generated).sum().item())


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
    input_ids, attention_mask = prepare_prompt(tokenizer, args.prompt)
    prompt_length = input_ids.shape[1]

    scenarios = [
        ("one-node baseline", False, 1.0),
        ("two nodes, always available", True, 0.0),
        (f"two nodes, dropout={args.peer_dropout:.2f}", True, args.peer_dropout),
    ]
    baseline: SwarmResult | None = None

    for label, enable_peer, peer_dropout in scenarios:
        torch.cuda.empty_cache()
        reset_cuda_peak_memory(torch)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            result = swarm_generate(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_token_id=tokenizer.mask_token_id,
                max_new_tokens=args.max_new_tokens,
                steps=args.steps,
                temperature=args.temperature,
                top_p=args.top_p,
                primary_seed=args.seed,
                peer_seed=args.seed + 1,
                enable_peer=enable_peer,
                peer_dropout=peer_dropout,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        text = decode_response(tokenizer, result.sequences, prompt_length)
        validate_generated_text(text, tokenizer.mask_token)
        peak_allocated, peak_reserved = cuda_peak_memory_gb(torch)

        if baseline is None:
            baseline = result
        differences = _token_differences(result, baseline, prompt_length)

        print(f"\n=== {label} ===")
        print(text)
        print("Statistics:")
        print(f"  elapsed: {elapsed:.2f} s")
        print(f"  active nodes per step: {result.active_nodes_per_step}")
        print(f"  optional-node proposals committed: {result.peer_wins}")
        print(f"  optional-node failures ignored: {result.peer_failures}")
        print(f"  output-token differences from baseline: {differences}")
        print(f"  PyTorch peak allocated VRAM: {peak_allocated:.2f} GiB")
        print(f"  PyTorch peak reserved VRAM: {peak_reserved:.2f} GiB")
        enforce_vram_limit(peak_reserved, args.max_vram_gb)


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
