import importlib.util
import unittest
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch


class TinyModel:
    device = "cpu"

    def __init__(self):
        self.forward_calls = 0

    def __call__(self, input_ids, attention_mask=None, token_indices=None):
        self.forward_calls += 1
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
    eos_token_id = 4

    def apply_chat_template(self, messages, **kwargs):
        return SimpleNamespace(
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
        )

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(token_id) for token_id in token_ids)


class ExperimentTests(unittest.TestCase):
    def test_comparison_sends_peer_scenarios_to_distinct_model(self):
        from experiment import run_comparison

        args = SimpleNamespace(
            prompt="test prompt", max_new_tokens=4, steps=4, temperature=0.8,
            top_p=0.95, seed=7, peer_dropout=0.0, max_vram_gb=16.0,
            remask_fraction=0.0,
        )
        primary, peer = TinyModel(), TinyModel()
        with patch("experiment.torch.cuda.is_available", return_value=False):
            list(run_comparison(args, primary, TinyTokenizer(), peer_model=peer))

        self.assertEqual(primary.forward_calls, 16)
        self.assertEqual(peer.forward_calls, 8)

    def test_comparison_streams_step_updates_and_four_results(self):
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
                "one-node, remask=5%",
            ],
        )
        self.assertEqual(len(progress), 16)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(result.text for result in results))
        self.assertEqual(results[0].token_differences, 0)
        self.assertEqual(results[0].active_nodes_per_step, (1, 1, 1, 1))
        self.assertEqual(results[1].active_nodes_per_step, (2, 2, 2, 2))
        self.assertIsNotNone(results[0].first_eos_position)
        self.assertGreaterEqual(results[0].first_eos_position, 0)
        self.assertLess(results[0].first_eos_position, 4)
        self.assertEqual(results[0].remasked_tokens, 0)
        self.assertLessEqual(results[3].remasked_tokens, 1)

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

    def test_scenario_elapsed_excludes_consumer_wait_between_step_events(self):
        from experiment import ExperimentProgress, ScenarioResult, run_comparison

        args = SimpleNamespace(
            prompt="test prompt", max_new_tokens=4, steps=4, temperature=0.8,
            top_p=0.95, seed=7, peer_dropout=0.5, max_vram_gb=8.0,
            remask_fraction=0.05,
        )
        started = time.perf_counter()
        with patch("experiment.torch.cuda.is_available", return_value=False):
            events = []
            for event in run_comparison(args, TinyModel(), TinyTokenizer()):
                events.append(event)
                if isinstance(event, ExperimentProgress):
                    time.sleep(0.03)
        wall_seconds = time.perf_counter() - started
        results = [event for event in events if isinstance(event, ScenarioResult)]
        sampler_seconds = sum(result.elapsed_seconds for result in results)

        self.assertEqual(len(results), 4)
        self.assertGreater(wall_seconds - sampler_seconds, 0.25)


if __name__ == "__main__":
    unittest.main()
