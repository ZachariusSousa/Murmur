"""UI-independent orchestration and records for Dream swarm comparisons."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import time

import torch

from run_dream import (
    CUDA_DEVICE,
    cuda_peak_memory_gb,
    decode_response,
    enforce_vram_limit,
    prepare_prompt,
    reset_cuda_peak_memory,
    validate_generated_text,
)
from swarm_sampler import StepTrace, SwarmResult, swarm_generate


@dataclass(frozen=True)
class ExperimentProgress:
    scenario: str
    trace: StepTrace


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    text: str
    elapsed_seconds: float
    active_nodes_per_step: tuple[int, ...]
    peer_commits: int
    peer_failures: int
    token_differences: int
    peak_allocated_gb: float
    peak_reserved_gb: float


def _cuda_synchronize_if_available() -> bool:
    if not torch.cuda.is_available():
        return False
    torch.cuda.synchronize()
    return True


def run_comparison(
    args: object, model: object, tokenizer: object
) -> Iterator[ExperimentProgress | ScenarioResult]:
    """Yield live step events followed by each completed scenario result."""
    device = getattr(model, "device", CUDA_DEVICE)
    input_ids, attention_mask = prepare_prompt(
        tokenizer, args.prompt, device=device
    )
    prompt_length = input_ids.shape[1]
    scenarios = (
        ("one-node baseline", False, 1.0),
        ("two nodes, always available", True, 0.0),
        (f"two nodes, dropout={args.peer_dropout:.2f}", True, args.peer_dropout),
    )
    baseline_tokens: torch.Tensor | None = None

    for scenario, enable_peer, peer_dropout in scenarios:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            reset_cuda_peak_memory(torch)
        _cuda_synchronize_if_available()
        started = time.perf_counter()
        result: SwarmResult | None = None

        for event in swarm_generate(
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
        ):
            if isinstance(event, StepTrace):
                yield ExperimentProgress(scenario=scenario, trace=event)
            else:
                result = event

        _cuda_synchronize_if_available()
        elapsed = time.perf_counter() - started
        if result is None:
            raise RuntimeError(f"{scenario} ended without a final sampler result")

        text = decode_response(tokenizer, result.sequences, prompt_length)
        validate_generated_text(text, tokenizer.mask_token)
        output_tokens = result.sequences[:, prompt_length:].detach().clone()
        if baseline_tokens is None:
            baseline_tokens = output_tokens
            token_differences = 0
        else:
            token_differences = int((output_tokens != baseline_tokens).sum().item())

        if torch.cuda.is_available():
            peak_allocated, peak_reserved = cuda_peak_memory_gb(torch)
        else:
            peak_allocated, peak_reserved = 0.0, 0.0
        enforce_vram_limit(peak_reserved, args.max_vram_gb)

        yield ScenarioResult(
            scenario=scenario,
            text=text,
            elapsed_seconds=elapsed,
            active_nodes_per_step=tuple(result.active_nodes_per_step),
            peer_commits=result.peer_wins,
            peer_failures=result.peer_failures,
            token_differences=token_differences,
            peak_allocated_gb=peak_allocated,
            peak_reserved_gb=peak_reserved,
        )
