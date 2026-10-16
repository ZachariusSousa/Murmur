"""Run Dream 7B Instruct as a 4-bit masked-diffusion language model."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence


MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
GIB = 1024**3
CUDA_DEVICE = "cuda:0"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Dream 7B Instruct in 4-bit NF4 on one NVIDIA GPU."
    )
    parser.add_argument("prompt", help="User prompt to denoise into a response.")
    parser.add_argument("--max-new-tokens", type=positive_int, default=64)
    parser.add_argument("--steps", type=positive_int, default=64)
    parser.add_argument("--temperature", type=non_negative_float, default=0.2)
    parser.add_argument("--top-p", type=probability, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-vram-gb",
        type=positive_float,
        default=8.0,
        help="Fail if PyTorch peak reserved VRAM exceeds this budget (default: 8).",
    )
    return parser


def validate_generated_text(text: str, mask_token: str | None) -> None:
    if not text.strip():
        raise RuntimeError("generation returned empty text")
    if mask_token and mask_token in text:
        raise RuntimeError("generation left an unresolved mask token in the output")


def enforce_vram_limit(peak_reserved_gb: float, limit_gb: float) -> None:
    if peak_reserved_gb > limit_gb:
        raise RuntimeError(
            f"PyTorch peak reserved VRAM {peak_reserved_gb:.2f} GiB exceeded "
            f"the {limit_gb:.2f} GiB budget"
        )


def reset_cuda_peak_memory(torch_module: object) -> None:
    torch_module.cuda.reset_peak_memory_stats()


def cuda_peak_memory_gb(torch_module: object) -> tuple[float, float]:
    return (
        torch_module.cuda.max_memory_allocated() / GIB,
        torch_module.cuda.max_memory_reserved() / GIB,
    )


def public_hugging_face_kwargs() -> dict[str, bool]:
    return {"token": False, "trust_remote_code": True}


def load_dream_model_and_tokenizer(
    torch_module: object,
    auto_model: object,
    auto_tokenizer: object,
    quantization_config_class: object,
) -> tuple[object, object]:
    """Load the shared 4-bit Dream components used by both experiments."""
    quantization = quantization_config_class(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_module.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    hub_kwargs = public_hugging_face_kwargs()
    tokenizer = auto_tokenizer.from_pretrained(MODEL_ID, **hub_kwargs)
    model = auto_model.from_pretrained(
        MODEL_ID,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        quantization_config=quantization,
        torch_dtype=torch_module.bfloat16,
        **hub_kwargs,
    ).eval()
    return model, tokenizer


def prepare_prompt(
    tokenizer: object, prompt: str, *, device: str = CUDA_DEVICE
) -> tuple[object, object]:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )
    return (
        inputs.input_ids.to(device),
        inputs.attention_mask.to(device),
    )


def decode_response(tokenizer: object, sequences: object, prompt_length: int) -> str:
    generated_ids = sequences[0, prompt_length:]
    text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=False)
    if tokenizer.eos_token:
        text = text.split(tokenizer.eos_token, 1)[0]
    return text.strip()


def run_generation(args: argparse.Namespace) -> None:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "dependencies are missing; run 'uv sync' in PowerShell"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but PyTorch cannot see an NVIDIA GPU; "
            "check the driver and the CUDA-enabled PyTorch installation"
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()
    reset_cuda_peak_memory(torch)

    print(f"Model: {MODEL_ID}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Loading 4-bit NF4 weights (the first run downloads the checkpoint)...")

    model, tokenizer = load_dream_model_and_tokenizer(
        torch, AutoModel, AutoTokenizer, BitsAndBytesConfig
    )
    input_ids, attention_mask = prepare_prompt(tokenizer, args.prompt)

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=args.steps,
            temperature=args.temperature,
            top_p=args.top_p,
            alg="entropy",
            alg_temp=0.0,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    text = decode_response(tokenizer, output.sequences, input_ids.shape[1])
    validate_generated_text(text, tokenizer.mask_token)

    peak_allocated_gb, peak_reserved_gb = cuda_peak_memory_gb(torch)

    print("\nGenerated text:")
    print(text)
    print("\nRun statistics:")
    print(f"  seed: {args.seed}")
    print(f"  diffusion steps: {args.steps}")
    print(f"  max new tokens: {args.max_new_tokens}")
    print(f"  elapsed: {elapsed:.2f} s")
    print(f"  PyTorch peak allocated VRAM: {peak_allocated_gb:.2f} GiB")
    print(f"  PyTorch peak reserved VRAM: {peak_reserved_gb:.2f} GiB")
    enforce_vram_limit(peak_reserved_gb, args.max_vram_gb)


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[argparse.Namespace], None] = run_generation,
) -> int:
    args = create_parser().parse_args(argv)
    try:
        runner(args)
    except (ImportError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
