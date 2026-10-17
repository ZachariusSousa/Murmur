import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch


class RunSwarmTests(unittest.TestCase):
    def test_parser_reuses_generation_defaults_and_adds_dropout(self):
        import run_swarm

        args = run_swarm.create_parser().parse_args(["Explain diffusion."])

        self.assertEqual(args.max_new_tokens, 64)
        self.assertEqual(args.steps, 64)
        self.assertEqual(args.temperature, 0.2)
        self.assertEqual(args.top_p, 0.95)
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.max_vram_gb, 16.0)
        self.assertEqual(args.peer_dropout, 0.5)
        self.assertEqual(args.remask_fraction, 0.05)

    def test_two_model_experiment_defaults_to_16_gib_budget(self):
        import run_swarm

        args = run_swarm.create_parser().parse_args(["Hello"])
        self.assertEqual(args.max_vram_gb, 16.0)

    def test_help_explains_two_model_vram_default(self):
        import run_swarm

        help_output = run_swarm.create_parser().format_help()
        self.assertIn("default: 16 GiB for two model copies", help_output)

    def test_loader_creates_two_model_instances_and_reuses_tokenizer(self):
        import run_swarm

        primary, peer, tokenizer = object(), object(), object()
        with patch(
            "run_swarm.load_dream_model_and_tokenizer",
            side_effect=[(primary, tokenizer), (peer, tokenizer)],
        ) as load:
            models = run_swarm.load_two_dream_nodes(
                object(), object(), object(), object()
            )

        self.assertIs(models[0], primary)
        self.assertIs(models[1], peer)
        self.assertIs(models[2], tokenizer)
        self.assertEqual(load.call_count, 2)
        self.assertNotEqual(id(models[0]), id(models[1]))

    def test_parser_accepts_remask_fraction_boundaries(self):
        import run_swarm

        disabled = run_swarm.create_parser().parse_args(
            ["Hello", "--remask-fraction", "0"]
        )
        enabled = run_swarm.create_parser().parse_args(
            ["Hello", "--remask-fraction", "0.2"]
        )
        self.assertEqual(disabled.remask_fraction, 0.0)
        self.assertEqual(enabled.remask_fraction, 0.2)

    def test_parser_rejects_out_of_range_remask_fraction(self):
        import run_swarm

        with self.assertRaises(SystemExit):
            run_swarm.create_parser().parse_args(
                ["Hello", "--remask-fraction", "1.01"]
            )

    def test_parser_accepts_dropout_boundaries(self):
        import run_swarm

        zero = run_swarm.create_parser().parse_args(["Hello", "--peer-dropout", "0"])
        one = run_swarm.create_parser().parse_args(["Hello", "--peer-dropout", "1"])

        self.assertEqual(zero.peer_dropout, 0.0)
        self.assertEqual(one.peer_dropout, 1.0)

    def test_parser_rejects_invalid_dropout(self):
        import run_swarm

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_swarm.create_parser().parse_args(
                ["Hello", "--peer-dropout", "1.01"]
            )

    def test_main_reports_concise_errors(self):
        import run_swarm

        def fail(_args):
            raise RuntimeError("CUDA is required")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = run_swarm.main(["Hello"], runner=fail)

        self.assertEqual(code, 1)
        self.assertEqual(stderr.getvalue().strip(), "error: CUDA is required")


if __name__ == "__main__":
    unittest.main()
