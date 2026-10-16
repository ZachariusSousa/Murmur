"""Small, local coordinator for Dream-style masked diffusion.

The denoising schedule mirrors the entropy sampler in Dream's Apache-2.0
``generation_utils.py``.  Node proposal and coordination are kept separate so
the same contract can later be moved across machines.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NodeProposal:
    node_name: str
    token_ids: torch.Tensor
    token_probabilities: torch.Tensor
    position_confidences: torch.Tensor


@dataclass(frozen=True)
class CombinedProposal:
    token_ids: torch.Tensor
    position_confidences: torch.Tensor
    peer_wins: int
    peer_winner_mask: torch.Tensor


@dataclass(frozen=True)
class SwarmResult:
    sequences: torch.Tensor
    active_nodes_per_step: list[int]
    peer_wins: int
    peer_failures: int


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = logits.clone()
    return filtered.scatter(
        -1,
        sorted_indices,
        sorted_logits.masked_fill(remove, float("-inf")),
    )


def sample_node_proposal(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
    node_name: str,
) -> NodeProposal:
    """Sample one candidate and its probability for every masked position."""
    sampling_logits = logits / temperature if temperature > 0 else logits
    sampling_logits = _top_p_filter(sampling_logits, top_p)
    probabilities = torch.softmax(sampling_logits, dim=-1)

    if temperature > 0:
        token_ids = torch.multinomial(
            probabilities, 1, generator=generator
        ).squeeze(-1)
    else:
        token_ids = probabilities.argmax(dim=-1)

    token_probabilities = probabilities.gather(
        -1, token_ids.unsqueeze(-1)
    ).squeeze(-1)
    entropy_confidence = torch.sum(
        probabilities * torch.log(probabilities.clamp_min(1e-10)), dim=-1
    )
    return NodeProposal(
        node_name=node_name,
        token_ids=token_ids,
        token_probabilities=token_probabilities,
        position_confidences=entropy_confidence,
    )


def combine_proposals(
    primary: NodeProposal, peers: list[NodeProposal]
) -> CombinedProposal:
    """Choose the highest-probability candidate at each masked position."""
    token_ids = primary.token_ids.clone()
    best_probabilities = primary.token_probabilities.clone()
    confidence_sum = primary.position_confidences.clone()
    peer_wins = 0
    peer_winner_mask = torch.zeros_like(token_ids, dtype=torch.bool)

    for peer in peers:
        if peer.token_ids.shape != token_ids.shape:
            raise ValueError("all node proposals must cover the same positions")
        wins = peer.token_probabilities > best_probabilities
        token_ids[wins] = peer.token_ids[wins]
        best_probabilities[wins] = peer.token_probabilities[wins]
        peer_winner_mask[wins] = True
        peer_wins += int(wins.sum().item())
        confidence_sum = confidence_sum + peer.position_confidences

    return CombinedProposal(
        token_ids=token_ids,
        position_confidences=confidence_sum / (1 + len(peers)),
        peer_wins=peer_wins,
        peer_winner_mask=peer_winner_mask,
    )


def _model_logits(model: object, x: torch.Tensor, attention_mask: object,
                  token_indices: torch.Tensor | None) -> torch.Tensor:
    return model(x, attention_mask, token_indices).logits


def swarm_generate(
    model: object,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    mask_token_id: int,
    max_new_tokens: int,
    steps: int,
    temperature: float,
    top_p: float,
    primary_seed: int,
    peer_seed: int,
    enable_peer: bool,
    peer_dropout: float = 0.0,
    peer_model: object | None = None,
) -> SwarmResult:
    """Run a batch-size-one Dream entropy schedule with optional proposals."""
    if input_ids.shape[0] != 1:
        raise ValueError("the local swarm experiment supports batch size 1 only")
    if not 0.0 <= peer_dropout <= 1.0:
        raise ValueError("peer_dropout must be between 0 and 1")

    prompt_length = input_ids.shape[1]
    max_length = prompt_length + max_new_tokens
    x = F.pad(input_ids, (0, max_new_tokens), value=mask_token_id)

    if attention_mask is not None and torch.any(attention_mask == 0):
        prepared_attention: object = F.pad(
            attention_mask, (0, max_new_tokens), value=1
        )
        token_indices = prepared_attention.long().cumsum(-1) - 1
        token_indices.masked_fill_(prepared_attention == 0, 1)
        prepared_attention = torch.logical_and(
            prepared_attention.unsqueeze(1).unsqueeze(-2),
            prepared_attention.unsqueeze(1).unsqueeze(-1),
        )
    else:
        prepared_attention = "full"
        token_indices = None

    primary_generator = torch.Generator(device=x.device).manual_seed(primary_seed)
    peer_generator = torch.Generator(device=x.device).manual_seed(peer_seed)
    dropout_generator = random.Random(primary_seed + 1_000_003)
    timesteps = torch.linspace(1.0, 0.001, steps + 1, device=x.device)
    actual_peer_model = model if peer_model is None else peer_model

    active_nodes_per_step: list[int] = []
    total_peer_wins = 0
    peer_failures = 0

    for step in range(steps):
        mask_index = x == mask_token_id
        if not torch.any(mask_index):
            active_nodes_per_step.extend([1] * (steps - step))
            break

        primary_logits = _model_logits(
            model, x, prepared_attention, token_indices
        )
        primary_logits = torch.cat(
            [primary_logits[:, :1], primary_logits[:, :-1]], dim=1
        )[mask_index]
        primary_logits = primary_logits.clone()
        primary_logits[:, mask_token_id] = float("-inf")
        primary = sample_node_proposal(
            primary_logits,
            temperature=temperature,
            top_p=top_p,
            generator=primary_generator,
            node_name="primary",
        )

        peers: list[NodeProposal] = []
        peer_requested = enable_peer and dropout_generator.random() >= peer_dropout
        if peer_requested:
            try:
                peer_logits = _model_logits(
                    actual_peer_model, x, prepared_attention, token_indices
                )
                peer_logits = torch.cat(
                    [peer_logits[:, :1], peer_logits[:, :-1]], dim=1
                )[mask_index]
                peer_logits = peer_logits.clone()
                peer_logits[:, mask_token_id] = float("-inf")
                peers.append(
                    sample_node_proposal(
                        peer_logits,
                        temperature=temperature,
                        top_p=top_p,
                        generator=peer_generator,
                        node_name="peer",
                    )
                )
            except Exception:
                # An optional contributor must never stop primary generation.
                peer_failures += 1

        active_nodes_per_step.append(1 + len(peers))
        combined = combine_proposals(primary, peers)

        remaining = int(mask_index.sum().item())
        if step < steps - 1:
            transfer_count = int(
                remaining * (1 - timesteps[step + 1] / timesteps[step])
            )
        else:
            transfer_count = remaining

        if transfer_count > 0:
            full_confidence = torch.full(
                x.shape,
                float("-inf"),
                device=x.device,
                dtype=primary_logits.dtype,
            )
            full_confidence[mask_index] = combined.position_confidences
            transfer_index = torch.topk(
                full_confidence, transfer_count, dim=-1
            ).indices
            candidates = torch.full_like(x, mask_token_id)
            candidates[mask_index] = combined.token_ids
            peer_winners = torch.zeros_like(x, dtype=torch.bool)
            peer_winners[mask_index] = combined.peer_winner_mask
            rows = torch.arange(x.shape[0], device=x.device).unsqueeze(1)
            rows = rows.expand_as(transfer_index)
            total_peer_wins += int(
                peer_winners[rows, transfer_index].sum().item()
            )
            x[rows, transfer_index] = candidates[rows, transfer_index]

    if x.shape[1] != max_length or not torch.equal(
        x[:, :prompt_length], input_ids
    ):
        raise RuntimeError("the coordinator changed prompt tokens")
    if torch.any(x[:, prompt_length:] == mask_token_id):
        raise RuntimeError("the coordinator left unresolved mask tokens")

    return SwarmResult(
        sequences=x,
        active_nodes_per_step=active_nodes_per_step,
        peer_wins=total_peer_wins,
        peer_failures=peer_failures,
    )
