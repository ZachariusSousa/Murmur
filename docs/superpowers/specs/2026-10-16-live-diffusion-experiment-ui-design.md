# Live Diffusion Experiment UI Design

## Purpose

Give one researcher a simple local browser interface for launching and observing
the existing Dream single-node and two-node coordination experiments. The UI
should make denoising behavior and experimental results easy to inspect, while
keeping its structure straightforward to adapt for a Hugging Face Space later.

The first measured comparison makes latency a core observation: on the RTX 4090
Laptop GPU, one node took 12.52 seconds, two sequential nodes took 24.92 seconds,
and the 50% peer-dropout run took 18.94 seconds. The UI must label the current
two-node simulation as sequential and display its latency alongside the
single-node baseline. Concurrent execution with two model copies in VRAM or
separate machines remains a future experiment.

## Scope

The app is for one person running experiments on their own machine. It will:

- launch the baseline, always-available peer, and configured peer-dropout runs;
- stream per-step progress to the browser while those runs execute;
- show the generated outputs and compare them with the baseline;
- report active nodes, peer-committed tokens, elapsed time, and peak PyTorch VRAM;
- use a restrained cyberpunk visual style: dark surfaces, readable typography,
  and a small number of neon accents.

It will not add networking between model nodes, accounts, storage, training,
model adapters, distributed scheduling, or production deployment setup.

## User experience

The page has a compact control area for prompt, output-token count, diffusion
steps, temperature, top-p, seed, VRAM limit, and peer dropout. A single launch
action starts the comparison. The page streams updates as each scenario runs
and shows three outputs together when available: baseline, two nodes without
dropout, and two nodes with the selected dropout rate.

The live view contains:

- a token-by-step view showing masked positions and which contributor's token
  was committed;
- a small timeline for masks remaining and active nodes;
- summary plots for peer commits and scenario elapsed time;
- final text outputs and memory measurements.

The page must state clearly that peer model calls are sequential in this
version. It should not imply that peer participation improves text quality;
proposal influence, output difference, elapsed time, and VRAM are measurements,
not quality scores.

## Application structure

Use Gradio Blocks as a thin browser layer, with a local Python entry point such
as `app.py`. Gradio supports local launch and provides a direct path to hosting
the same app in a Hugging Face Space. The UI calls the existing model-loading
helpers and swarm sampler rather than duplicating diffusion logic.

The sampler will expose a small per-step progress callback or event record. Each
record should contain the scenario name, step index, masks remaining, active
node count, and peer tokens committed at that step. Scenario completion adds
the output, elapsed time, token differences from baseline, and peak VRAM. The UI
consumes these records to refresh its status, grid, and plots.

Load the model lazily on the first launch and reuse it for subsequent runs while
the local app remains open. Permit one experiment at a time so concurrent UI
submissions cannot compete for the same GPU memory. The Gradio layer should
remain replaceable; experiment state and event records should be plain Python
data rather than UI-specific objects.

## Failure behavior

The primary node remains the required contributor. If the optional peer fails,
the sampler records that failure and continues with the primary proposal. If
model loading or the primary run fails, the UI displays a concise error and
leaves the controls available for another attempt. The app reports the existing
PyTorch-tracked VRAM numbers and their measurement scope.

## Verification

- Keep the existing command-line experiments operational.
- Verify the UI starts locally and its launch control accepts the same settings
  as the CLI.
- Verify the browser receives step events during generation, not only a final
  result.
- Verify all three scenarios complete and their outputs, peer commits,
  differences, timing, and VRAM are displayed from the recorded run data.
- Verify peer dropout and peer failure still permit generation to continue.
- Confirm the app structure can be launched as a Hugging Face Space without
  changing the experiment contract.

## Research caveat

The current two nodes share identical weights. Earlier tests showed that the
peer can change committed tokens, especially at higher sampling temperature,
but did not establish improved output quality or learned specialization. The
observability view should present evidence neutrally so later experiments with
separate adapters, model copies, or machines can be compared honestly.
