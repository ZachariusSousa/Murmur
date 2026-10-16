# Live Diffusion Experiment UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight local Gradio page that launches the existing Dream comparison and streams denoising progress, outputs, latency, and VRAM.

**Architecture:** Keep diffusion logic in `swarm_sampler.py`, expose plain per-step events, and orchestrate the three scenarios in a UI-independent `experiment.py`. `app.py` will adapt that event stream to Gradio Blocks, lazily cache one model, and keep experiment concurrency at one. The same app entry point and experiment contract should remain suitable for a later Hugging Face Space.

**Tech Stack:** Python 3.11, PyTorch, Transformers, Gradio `>=6,<7`, pandas-backed Gradio plots, unittest, uv. See the official [LinePlot](https://gradio.app/docs/gradio/lineplot) and [queue](https://gradio.app/main/docs/gradio/blocks) documentation for the UI primitives.

**Spec:** `docs/superpowers/specs/2026-10-16-live-diffusion-experiment-ui-design.md`

## Global Constraints

- The app is for one person running experiments on their own machine.
- The current two-node simulation calls peer models sequentially and must be labeled as sequential.
- Load the model lazily on the first launch and reuse it for subsequent runs while the local app remains open.
- Permit one experiment at a time so concurrent UI submissions cannot compete for the same GPU memory.
- Use Gradio Blocks as a thin browser layer; experiment state and event records remain plain Python data.
- Proposal influence, output difference, elapsed time, and VRAM are measurements, not quality scores.
- Keep networking between model nodes, accounts, storage, training, adapters, distributed scheduling, and production deployment setup out of scope.

## Review Focus

- A failure while loading the model must display a concise error and leave controls available: test the UI generator with an injected loader that raises.
- A peer exception or peer dropout must preserve primary generation and continue streaming: test the streamed scenario event sequence with the existing failing-peer fixture.
- Repeated or concurrent launches must not run two GPU experiments at once: test the launch concurrency configuration and the cached loader call count.
- A prompt or setting rejected by the current CLI validation must fail before generation and be shown as a concise UI error: test empty prompt and invalid dropout through the shared settings validation.
- A run that finishes early or emits no peer commits must still render valid progress plots and final output: test a short deterministic mock-model run with zero peer commits.

---

## File Structure

- `swarm_sampler.py` owns diffusion sampling and emits serializable step events alongside the final tensor result.
- `experiment.py` owns the three-scenario comparison and plain result/event records; it does not import Gradio.
- `run_swarm.py` remains the CLI adapter and consumes the comparison records for console output.
- `app.py` owns model caching, Gradio Blocks, live presentation, styling, and local/Hugging Face Space launch behavior.
- `pyproject.toml` and `uv.lock` declare and lock Gradio.
- `tests/test_swarm_sampler.py`, `tests/test_experiment.py`, and `tests/test_app.py` cover sampler events, scenario metrics, and the browser adapter without loading Dream weights.
- `README.md` documents the PowerShell launch command, controls, sequential timing caveat, and how to start the app for local use.

### Task 1: Emit per-step denoising events

**Files:**
- Modify: `swarm_sampler.py`
- Test: `tests/test_swarm_sampler.py`

**Interfaces:**
- Produces `TokenCommit(position: int, token_id: int, source: str)` where `source` is `"primary"` or `"peer"`.
- Produces `StepTrace(step: int, masks_remaining: int, active_nodes: int, committed_tokens: tuple[TokenCommit, ...])`.
- Changes `swarm_generate(...)` to an iterator yielding one `StepTrace` after each diffusion step and the final `SwarmResult` last.

- [ ] **Step 1: Add a failing trace test** asserting that a four-step run emits four ordered `StepTrace` records and one final `SwarmResult`; each record has the expected active node count, non-increasing masks remaining, and committed token positions with the correct contributor source.
- [ ] **Step 2: Run the focused test and confirm it fails** because `swarm_generate` currently returns only `SwarmResult` and emits no trace.

Run in PowerShell: `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_swarm_sampler.py" -v`

Expected: FAIL because step trace records are not emitted.
- [ ] **Step 3: Implement the trace records and iterator** in `swarm_sampler.py`. Emit commits only for positions actually transferred during that step, derive their source from the coordinator's peer-winner mask, and keep the final `SwarmResult` as the last yielded item.
- [ ] **Step 4: Update sampler tests** to consume the iterator and retain coverage for independent sampling, one-node mode, exact full-dropout baseline equivalence, zero dropout, peer failure, prompt preservation, and no unresolved masks.
- [ ] **Step 5: Run the focused sampler tests and confirm they pass.**
- [ ] **Step 6: Commit** with message `feat: stream diffusion step traces`.

### Task 2: Extract UI-independent experiment orchestration

**Files:**
- Create: `experiment.py`
- Modify: `run_swarm.py`
- Test: `tests/test_experiment.py`
- Modify: `tests/test_run_swarm.py`

**Interfaces:**
- Consumes `StepTrace` and `SwarmResult` from Task 1.
- Produces `ExperimentProgress(scenario: str, trace: StepTrace)` and `ScenarioResult(scenario: str, text: str, elapsed_seconds: float, active_nodes_per_step: tuple[int, ...], peer_commits: int, peer_failures: int, token_differences: int, peak_allocated_gb: float, peak_reserved_gb: float)`.
- Produces `run_comparison(args, model, tokenizer) -> Iterator[ExperimentProgress | ScenarioResult]`, yielding step progress during each scenario and a `ScenarioResult` at its completion.

- [ ] **Step 1: Add a failing orchestration test** using the tiny fake model and tokenizer; assert that `run_comparison` yields progress before each of three `ScenarioResult` records and correctly reports baseline differences, committed peer tokens, elapsed time, and active node counts.
- [ ] **Step 2: Run the focused test and confirm it fails** because `experiment.py` does not exist.

Run in PowerShell: `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_experiment.py" -v`

Expected: FAIL because the comparison iterator is not implemented.
- [ ] **Step 3: Implement `run_comparison`** in `experiment.py`, preserving the current scenario order and seeds: one-node baseline, two nodes with zero dropout, then two nodes with the configured dropout. Yield each step as it arrives, and include per-scenario VRAM readings and token differences in the completion record.
- [ ] **Step 4: Adapt `run_swarm.py`** to consume `run_comparison` and preserve its current command-line options and printed metrics.
- [ ] **Step 5: Run the focused orchestration and CLI tests and confirm they pass.**
- [ ] **Step 6: Commit** with message `refactor: expose comparison events for the UI`.

### Task 3: Build the lightweight live Gradio page

**Files:**
- Create: `app.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `tests/test_app.py`
- Modify: `README.md`

**Interfaces:**
- Consumes `run_comparison(args, model, tokenizer)` from Task 2.
- Produces `build_app(model_loader=...) -> gr.Blocks` so UI structure and model loading can be tested independently.
- Produces a launch handler that streams updated status, token-by-step view, masks-remaining and active-node plots, and three final scenario summaries as `ExperimentProgress` and `ScenarioResult` records arrive.

- [ ] **Step 1: Add failing app tests** asserting that the model loader is called lazily once and reused, invalid settings return a concise error without loading the model, and the launch event is limited to one concurrent experiment.
- [ ] **Step 2: Run the focused test and confirm it fails** because `app.py` does not exist.

Run in PowerShell: `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_app.py" -v`

Expected: FAIL because the app factory and launch handler are not implemented.
- [ ] **Step 3: Add `gradio>=6,<7`** to `pyproject.toml`, resolve `uv.lock`, and keep the existing Python 3.11 and CUDA model dependencies unchanged.
- [ ] **Step 4: Implement the app factory and launch handler** in `app.py`. Cache one model/tokenizer pair lazily, use Gradio's queued event with `concurrency_limit=1`, and stream the experiment iterator directly to browser outputs. Use built-in `gr.LinePlot` components for the step timelines and compact custom CSS for the dark cyberpunk palette.
- [ ] **Step 5: Add a local entry point** that binds to `127.0.0.1` for local execution and uses the Hugging Face Space environment when present; do not enable public sharing for local runs.
- [ ] **Step 6: Update the README** with the exact PowerShell command `uv run python .\app.py`, the local bootstrap executable form already used in this workspace, a short explanation of the displayed metrics, and the sequential latency caveat.
- [ ] **Step 7: Run all unit tests, then start the app locally** with `uv run python .\app.py`; confirm the page loads before any experiment starts and the lazy model loader has not run.
- [ ] **Step 8: Commit** with message `feat: add live diffusion experiment UI`.

## Self-Review

- **Spec coverage:** launch controls, lazy model reuse, one-at-a-time execution, streamed step events, token grid, simple plots, three final outputs, sequential labeling, peer failure behavior, concise errors, local launch, and Hugging Face Space compatibility are assigned to Tasks 1–3.
- **Step scan:** each task begins with a failing unit test, then adds one implementation unit and a focused verification. The app uses the event iterator directly, avoiding a separate polling thread or new service.
- **Type consistency:** `swarm_generate` emits `StepTrace` then `SwarmResult`; `run_comparison` converts those into `ExperimentProgress` then `ScenarioResult`; the UI consumes those exact event types.
- **Review focus:** each of the five listed input/failure cases is assigned to a named test in Task 2 or Task 3.
- **Proportion:** three tasks cover the sampler contract, scenario orchestration, and browser adapter; no networking, persistence, deployment pipeline, or training subsystem is included.
