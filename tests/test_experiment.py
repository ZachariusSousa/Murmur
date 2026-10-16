import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


class TinyModel:
    device = "cpu"

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


class TinyTokenizer:
    mask_token_id = 5
    mask_token = "<mask>"
    eos_token = None

    def apply_chat_template(self, messages, **kwargs):
        return SimpleNamespace(
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
        )

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(token_id) for token_id in token_ids)


class ExperimentTests(unittest.TestCase):
    def test_comparison_streams_step_updates_and_three_results(self):
        module_spec = importlib.util.find_spec("experiment")
        self.assertIsNotNone(
            module_spec,
            "experiment.run_comparison must stream comparison records",
        )

        from experiment import ExperimentProgress, ScenarioResult, run_comparison

        args = SimpleNamespace(
            prompt="test prompt",
            max_new_tokens=4,
            steps=4,
            temperature=0.8,
            top_p=0.95,
            seed=7,
            peer_dropout=0.5,
            max_vram_gb=8.0,
        )
        with patch("experiment.torch.cuda.is_available", return_value=False):
            events = list(run_comparison(args, TinyModel(), TinyTokenizer()))

        results = [event for event in events if isinstance(event, ScenarioResult)]
        progress = [event for event in events if isinstance(event, ExperimentProgress)]

        self.assertEqual(
            [result.scenario for result in results],
            [
                "one-node baseline",
                "two nodes, always available",
                "two nodes, dropout=0.50",
            ],
        )
        self.assertEqual(len(progress), 12)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.text for result in results))
        self.assertEqual(results[0].token_differences, 0)
        self.assertEqual(results[0].active_nodes_per_step, (1, 1, 1, 1))
        self.assertEqual(results[1].active_nodes_per_step, (2, 2, 2, 2))

        for result in results:
            scenario_progress = [
                event.trace
                for event in progress
                if event.scenario == result.scenario
            ]
            peer_commits = sum(
                token.source == "peer"
                for trace in scenario_progress
                for token in trace.committed_tokens
            )
            self.assertEqual(result.peer_commits, peer_commits)
            self.assertGreaterEqual(result.elapsed_seconds, 0.0)
            self.assertGreaterEqual(result.peak_allocated_gb, 0.0)
            self.assertGreaterEqual(result.peak_reserved_gb, 0.0)


if __name__ == "__main__":
    unittest.main()
