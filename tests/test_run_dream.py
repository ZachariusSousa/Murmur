import importlib
import io
import sys
import unittest
from contextlib import redirect_stderr


class RunDreamTests(unittest.TestCase):
    def test_parser_uses_reproducible_low_memory_defaults(self):
        run_dream = importlib.import_module("run_dream")

        args = run_dream.create_parser().parse_args(["Explain diffusion."])

        self.assertEqual(args.prompt, "Explain diffusion.")
        self.assertEqual(args.max_new_tokens, 64)
        self.assertEqual(args.steps, 64)
        self.assertEqual(args.temperature, 0.2)
        self.assertEqual(args.top_p, 0.95)
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.max_vram_gb, 8.0)

    def test_parser_accepts_all_experiment_controls(self):
        run_dream = importlib.import_module("run_dream")

        args = run_dream.create_parser().parse_args(
            [
                "Hello",
                "--max-new-tokens", "32",
                "--steps", "16",
                "--temperature", "0.4",
                "--top-p", "0.8",
                "--seed", "7",
                "--max-vram-gb", "6.5",
            ]
        )

        self.assertEqual(args.max_new_tokens, 32)
        self.assertEqual(args.steps, 16)
        self.assertEqual(args.temperature, 0.4)
        self.assertEqual(args.top_p, 0.8)
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.max_vram_gb, 6.5)

    def test_parser_rejects_non_positive_counts(self):
        run_dream = importlib.import_module("run_dream")
        stderr = io.StringIO()

        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            run_dream.create_parser().parse_args(
                ["Hello", "--max-new-tokens", "0"]
            )

        self.assertIn("positive integer", stderr.getvalue())

    def test_validate_generated_text_rejects_empty_and_masked_output(self):
        run_dream = importlib.import_module("run_dream")

        with self.assertRaisesRegex(RuntimeError, "empty text"):
            run_dream.validate_generated_text("   ", "<mask>")
        with self.assertRaisesRegex(RuntimeError, "unresolved mask"):
            run_dream.validate_generated_text("Answer: <mask>", "<mask>")

    def test_vram_limit_raises_when_peak_exceeds_budget(self):
        run_dream = importlib.import_module("run_dream")

        with self.assertRaisesRegex(RuntimeError, "exceeded"):
            run_dream.enforce_vram_limit(8.01, 8.0)

        run_dream.enforce_vram_limit(8.0, 8.0)

    def test_main_reports_a_concise_error(self):
        run_dream = importlib.import_module("run_dream")

        def fail(_args):
            raise RuntimeError("CUDA is required")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = run_dream.main(["Hello"], runner=fail)

        self.assertEqual(exit_code, 1)
        self.assertEqual(stderr.getvalue().strip(), "error: CUDA is required")

    def test_cuda_memory_helpers_accept_the_windows_cuda_device(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        run_dream = importlib.import_module("run_dream")
        run_dream.reset_cuda_peak_memory(torch)
        allocated_gb, reserved_gb = run_dream.cuda_peak_memory_gb(torch)

        self.assertGreaterEqual(allocated_gb, 0.0)
        self.assertGreaterEqual(reserved_gb, 0.0)

    def test_public_model_download_does_not_send_stored_credentials(self):
        run_dream = importlib.import_module("run_dream")

        kwargs = run_dream.public_hugging_face_kwargs()

        self.assertIs(kwargs["token"], False)
        self.assertIs(kwargs["trust_remote_code"], True)


if __name__ == "__main__":
    unittest.main()
