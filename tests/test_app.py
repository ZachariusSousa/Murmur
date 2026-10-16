import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from swarm_sampler import TokenCommit


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        import torch
        return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))

    def convert_ids_to_tokens(self, token_id):
        return f"tok{token_id}"


class AppTests(unittest.TestCase):
    def test_loader_is_lazy_and_cached(self):
        import app

        loader = Mock(return_value=(object(), object()))
        cache = app.ModelCache(loader)
        self.assertEqual(loader.call_count, 0)
        self.assertIs(cache.get()[0], cache.get()[0])
        self.assertEqual(loader.call_count, 1)

    def test_invalid_settings_return_error_without_loading(self):
        import app

        loader = Mock(side_effect=AssertionError("must not load"))
        launch = app.create_launch_handler(loader)
        outputs = list(launch("  ", 16, 16, 0.2, 0.95, 0, 8, 0.5))
        self.assertEqual(len(outputs), 1)
        self.assertIn("prompt", outputs[0][0].lower())
        self.assertEqual(loader.call_count, 0)

    def test_invalid_dropout_returns_error_without_loading(self):
        import app

        loader = Mock(side_effect=AssertionError("must not load"))
        launch = app.create_launch_handler(loader)
        outputs = list(launch("valid", 16, 16, 0.2, 0.95, 0, 8, 1.5))
        self.assertIn("dropout", outputs[0][0].lower())
        self.assertEqual(loader.call_count, 0)

    def test_model_loading_failure_is_concise_and_recoverable(self):
        import app

        loader = Mock(side_effect=RuntimeError("CUDA initialization failed"))
        launch = app.create_launch_handler(loader)
        outputs = list(launch("valid", 16, 16, 0.2, 0.95, 0, 8, 0.5))
        self.assertEqual(len(outputs), 1)
        self.assertIn("CUDA initialization failed", outputs[0][0])
        self.assertEqual(loader.call_count, 1)

    def test_launch_streams_updates_and_uses_cached_loader(self):
        import app
        from experiment import ExperimentProgress, ScenarioResult
        from swarm_sampler import StepTrace

        model, tokenizer = object(), TinyTokenizer()
        loader = Mock(return_value=(model, tokenizer))
        events = [
            ExperimentProgress(
                "one-node baseline",
                StepTrace(0, 1, 1, (TokenCommit(2, 4, "peer"),)),
            ),
            ScenarioResult("one-node baseline", "hello", 1.0, (1,), 0, 0, 0, 1, 1),
        ]
        with patch("app.run_comparison", return_value=iter(events)) as comparison:
            launch = app.create_launch_handler(loader)
            updates = list(launch("Say hello", 1, 1, 0.2, 0.95, 0, 8, 0.5))
            list(launch("Say hello", 1, 1, 0.2, 0.95, 0, 8, 0.5))
        self.assertGreaterEqual(len(updates), 2)
        self.assertIn("0: tok4", updates[0][1])
        self.assertEqual(list(updates[0][2].columns), ["step", "masks", "scenario"])
        self.assertEqual(list(updates[0][3].columns), ["step", "active nodes", "scenario"])
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(comparison.call_count, 2)

    def test_app_uses_single_experiment_concurrency_limit(self):
        import app

        self.assertEqual(app.EXPERIMENT_CONCURRENCY_LIMIT, 1)


if __name__ == "__main__":
    unittest.main()
