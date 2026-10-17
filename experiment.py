"""UI-independent orchestration and records for Dream swarm comparisons."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

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
    remasked_tokens: int = 0
    first_eos_position: int | None = None


def _cuda_synchronize_if_available() -> bool:
    if not torch.cuda.is_available():
        return False
    torch.cuda.synchronize()
    return True


def run_comparison(
    args: object, model: object, tokenizer: object, *, peer_model: object | None = None
) -> Iterator[ExperimentProgress | ScenarioResult]:
    """Yield progress/results while optionally coordinating two model copies."""
    device = getattr(model, "device", CUDA_DEVICE)
    input_ids, attention_mask = prepare_prompt(
        tokenizer, args.prompt, device=device
    )
    prompt_length = input_ids.shape[1]
    remask_fraction = getattr(args, "remask_fraction", 0.05)
    scenarios = (
        ("one-node baseline", False, 1.0, 0.0),
        ("two nodes, always available", True, 0.0, 0.0),
        (f"two nodes, dropout={args.peer_dropout:.2f}", True, args.peer_dropout, 0.0),
        (f"one-node, remask={remask_fraction:.0%}", False, 1.0, remask_fraction),
    )
    baseline_tokens: torch.Tensor | None = None

    for scenario, enable_peer, peer_dropout, scenario_remask_fraction in scenarios:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            reset_cuda_peak_memory(torch)
        _cuda_synchronize_if_available()
        scenario_elapsed = 0.0
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
            remask_fraction=scenario_remask_fraction,
            peer_model=peer_model,
        ):
            if isinstance(event, StepTrace):
                scenario_elapsed += event.elapsed_seconds
                yield ExperimentProgress(scenario=scenario, trace=event)
            else:
                result = event

        _cuda_synchronize_if_available()
        if result is None:
            raise RuntimeError(f"{scenario} ended without a final sampler result")

        text = decode_response(tokenizer, result.sequences, prompt_length)
        validate_generated_text(text, tokenizer.mask_token)
        output_tokens = result.sequences[:, prompt_length:].detach().clone()
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        eos_positions = (
            torch.nonzero(output_tokens[0] == eos_token_id, as_tuple=False).flatten()
            if eos_token_id is not None
            else torch.empty(0, dtype=torch.long, device=output_tokens.device)
        )
        first_eos_position = (
            int(eos_positions[0].item()) if eos_positions.numel() else None
        )
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
            elapsed_seconds=scenario_elapsed,
            active_nodes_per_step=tuple(result.active_nodes_per_step),
            peer_commits=result.peer_wins,
            peer_failures=result.peer_failures,
            token_differences=token_differences,
            peak_allocated_gb=peak_allocated,
            peak_reserved_gb=peak_reserved,
            remasked_tokens=result.remasked_tokens,
            first_eos_position=first_eos_position,
        )
