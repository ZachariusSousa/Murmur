import io
import unittest
from contextlib import redirect_stderr


class RunSwarmTests(unittest.TestCase):
    def test_parser_reuses_generation_defaults_and_adds_dropout(self):
        import run_swarm

        args = run_swarm.create_parser().parse_args(["Explain diffusion."])

        self.assertEqual(args.max_new_tokens, 64)
        self.assertEqual(args.steps, 64)
        self.assertEqual(args.temperature, 0.2)
        self.assertEqual(args.top_p, 0.95)
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.max_vram_gb, 8.0)
        self.assertEqual(args.peer_dropout, 0.5)

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
