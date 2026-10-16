# Dream 7B under 8GB VRAM

This is a deliberately small Windows/NVIDIA experiment. It loads
[`Dream-org/Dream-v0-Instruct-7B`](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B)
with 4-bit NF4 weights, calls Dream's native masked-diffusion generator, and
fails if PyTorch's peak reserved GPU memory exceeds the requested budget.

The first run downloads the official BF16 checkpoint (about 16GB on disk).
Quantization happens while the model loads and reduces its GPU footprint; it
does not shrink the Hugging Face download or cache.

## Setup in PowerShell

From this directory, install `uv` once:

```powershell
winget install --id astral-sh.uv --exact
```

Close and reopen PowerShell if `uv` is not immediately found. Then let `uv`
install Python 3.11 and the pinned CUDA environment:

```powershell
cd C:\Murmur
uv python install 3.11
uv sync
```

The NVIDIA driver may expose a newer CUDA version than PyTorch uses. That is
normal: this project installs PyTorch's CUDA 12.4 wheel, which works with a
newer compatible NVIDIA driver.

## Run it

The first smoke test uses deterministic settings, 64 output tokens, and 64
diffusion steps:

```powershell
uv run python .\run_dream.py "Explain masked diffusion language models in two sentences."
```

Try a second prompt or alter the experiment controls:

```powershell
uv run python .\run_dream.py "Write a four-line poem about parallel thought." --steps 32 --max-new-tokens 32 --seed 7
```

Show every option:

```powershell
uv run python .\run_dream.py --help
```

Run the lightweight unit tests without downloading the model:

```powershell
uv run python -m unittest discover -s tests -v
```

## Run the two-node coordination experiment

This reuses one loaded model to simulate two independent nodes sequentially.
Both see the same partially masked text and sample with different seeds. At
each masked position, the coordinator keeps the available proposal with the
higher token probability. It automatically compares:

1. the primary node alone;
2. both nodes on every diffusion step; and
3. both nodes with the peer absent on 50% of steps by default.

Use the local `uv` executable that is already installed in this project:

```powershell
cd C:\Murmur
.\.uv-bootstrap\Scripts\uv.exe run python .\run_swarm.py "Explain diffusion language models briefly."
```

For a faster first check, reduce both the output length and step count:

```powershell
.\.uv-bootstrap\Scripts\uv.exe run python .\run_swarm.py "Explain diffusion language models briefly." --max-new-tokens 16 --steps 16
```

Because the two nodes currently have identical weights, the conservative
default temperature can produce the same final answer. To make independent
proposals visibly affect this coordination experiment, try:

```powershell
.\.uv-bootstrap\Scripts\uv.exe run python .\run_swarm.py "Explain diffusion language models briefly." --max-new-tokens 16 --steps 16 --temperature 0.8
```

Change `--peer-dropout` from `0` (peer always present) to `1` (peer always
absent). The script prints generated text, runtime, the active-node count for
every step, peer proposals actually committed to the sequence, output-token
differences from the baseline, and peak PyTorch VRAM for each run.

The two nodes currently share weights. This proves the coordinator, independent
sampling, and graceful dropout behavior; it does not yet prove learned
specialization or better output quality.

## What the memory number means

The script reports PyTorch peak **allocated** and **reserved** VRAM from the
start of model loading through generation. The 8GB gate uses peak reserved
memory. NVIDIA driver allocations and memory used by unrelated applications
are not included in that number.

To verify that the gate itself fails, choose an intentionally tiny budget:

```powershell
uv run python .\run_dream.py "Say hello." --max-new-tokens 8 --steps 8 --max-vram-gb 0.1
if ($LASTEXITCODE -eq 0) { throw "Expected the VRAM gate to fail" }
```

## Troubleshooting

- **Interrupted model download:** run the same command again. Hugging Face
  resumes cached downloads.
- **`CUDA is required`:** run `nvidia-smi`, update the NVIDIA driver if it
  fails, then run `uv sync --reinstall-package torch`.
- **bitsandbytes CUDA error:** confirm you are on 64-bit Windows with a recent
  NVIDIA driver, then run `uv sync --reinstall-package bitsandbytes`.
- **Out of memory:** close other GPU applications and retry with shorter output
  and fewer steps. The model weights dominate usage, so the savings are
  limited.
- **Hugging Face access error:** this public-model prototype deliberately uses
  anonymous downloads, so stale saved credentials are ignored. Retry first;
  authentication is not currently required for this checkpoint.

This milestone intentionally has no CPU path, server, UI, training code, or
distributed contributor abstraction.
