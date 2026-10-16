import unittest
from types import SimpleNamespace

import torch


class ProposalTests(unittest.TestCase):
    def test_node_sampling_is_reproducible_but_independent(self):
        from swarm_sampler import sample_node_proposal

        logits = torch.zeros((32, 4))

        first = sample_node_proposal(
            logits,
            temperature=1.0,
            top_p=1.0,
            generator=torch.Generator().manual_seed(11),
            node_name="primary",
        )
        replay = sample_node_proposal(
            logits,
            temperature=1.0,
            top_p=1.0,
            generator=torch.Generator().manual_seed(11),
            node_name="primary",
        )
        peer = sample_node_proposal(
            logits,
            temperature=1.0,
            top_p=1.0,
            generator=torch.Generator().manual_seed(12),
            node_name="peer",
        )

        self.assertTrue(torch.equal(first.token_ids, replay.token_ids))
        self.assertFalse(torch.equal(first.token_ids, peer.token_ids))

    def test_coordinator_uses_highest_probability_candidate(self):
        from swarm_sampler import NodeProposal, combine_proposals

        primary = NodeProposal(
            node_name="primary",
            token_ids=torch.tensor([1, 1]),
            token_probabilities=torch.tensor([0.8, 0.2]),
            position_confidences=torch.tensor([-0.1, -0.2]),
        )
        peer = NodeProposal(
            node_name="peer",
            token_ids=torch.tensor([2, 2]),
            token_probabilities=torch.tensor([0.4, 0.9]),
            position_confidences=torch.tensor([-0.3, -0.4]),
        )

        combined = combine_proposals(primary, [peer])

        self.assertEqual(combined.token_ids.tolist(), [1, 2])
        self.assertEqual(combined.peer_wins, 1)
        torch.testing.assert_close(
            combined.position_confidences, torch.tensor([-0.2, -0.3])
        )


class TinyModel:
    def __call__(self, input_ids, attention_mask=None, token_indices=None):
        batch, length = input_ids.shape
        logits = torch.zeros((batch, length, 6), device=input_ids.device)
        logits[..., 0] = 0.2
        logits[..., 1] = 0.5
        logits[..., 2] = 0.8
        logits[..., 3] = 1.1
        logits[..., 4] = 1.4
        logits[..., 5] = -100.0
        return SimpleNamespace(logits=logits)


class FailingPeer:
    def __call__(self, input_ids, attention_mask=None, token_indices=None):
        raise RuntimeError("peer disappeared")


class SwarmGenerationTests(unittest.TestCase):
    def setUp(self):
        self.input_ids = torch.tensor([[1, 2]])
        self.attention_mask = torch.ones_like(self.input_ids)
        self.common = dict(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            mask_token_id=5,
            max_new_tokens=4,
            steps=4,
            temperature=0.8,
            top_p=0.95,
            primary_seed=7,
            peer_seed=8,
        )

    def test_one_node_completes_without_peer_and_preserves_prompt(self):
        from swarm_sampler import swarm_generate

        result = swarm_generate(TinyModel(), enable_peer=False, **self.common)

        self.assertEqual(result.active_nodes_per_step, [1, 1, 1, 1])
        self.assertEqual(result.peer_wins, 0)
        self.assertEqual(result.sequences[0, :2].tolist(), [1, 2])
        self.assertNotIn(5, result.sequences[0, 2:].tolist())

    def test_full_peer_dropout_exactly_matches_one_node(self):
        from swarm_sampler import swarm_generate

        baseline = swarm_generate(TinyModel(), enable_peer=False, **self.common)
        dropped = swarm_generate(
            TinyModel(), enable_peer=True, peer_dropout=1.0, **self.common
        )

        self.assertTrue(torch.equal(baseline.sequences, dropped.sequences))
        self.assertEqual(dropped.active_nodes_per_step, [1, 1, 1, 1])

    def test_zero_dropout_invokes_both_nodes_every_step(self):
        from swarm_sampler import swarm_generate

        result = swarm_generate(
            TinyModel(), enable_peer=True, peer_dropout=0.0, **self.common
        )

        self.assertEqual(result.active_nodes_per_step, [2, 2, 2, 2])
        self.assertGreater(result.peer_wins, 0)

    def test_failed_peer_is_ignored(self):
        from swarm_sampler import swarm_generate

        baseline = swarm_generate(TinyModel(), enable_peer=False, **self.common)
        with_failure = swarm_generate(
            TinyModel(),
            enable_peer=True,
            peer_dropout=0.0,
            peer_model=FailingPeer(),
            **self.common,
        )

        self.assertTrue(torch.equal(baseline.sequences, with_failure.sequences))
        self.assertEqual(with_failure.active_nodes_per_step, [1, 1, 1, 1])
        self.assertEqual(with_failure.peer_failures, 4)


if __name__ == "__main__":
    unittest.main()
