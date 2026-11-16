# Eardrum

Polyphonic piano transcription with a dual-head CRNN: one head predicts note
onsets, the other predicts sustained note frames conditioned on the onset
probabilities. A single-head baseline is included for the ablation.

## What it does

- Trains on real solo-piano recordings from MAESTRO v3.0.0.
- Uses a fixed tensor contract: 16 kHz mono, 4 s chunks (64000 samples),
  log-mel `(128, 126)`, 88 keys (MIDI 21–108), 31.25 frames/s.
- Evaluates with `mir_eval` note/onset/offset metrics on held-out tracks.
- Ships a Gradio app and a notebook with piano-roll visualization, threshold
  sliders, synthesized playback, and MIDI download.

## Install

```bash
uv sync
```

PyTorch resolves from the CPU-only index declared in `pyproject.toml`, so no
CUDA wheels are downloaded.

## Data

```bash
uv run python -m eardrum.dataset --download
```

This fetches the official MAESTRO v3.0.0 metadata JSON, downloads the six
shortest training tracks, two validation tracks, and two test tracks from the
pinned per-file mirror `ddPn08/maestro-v3.0.0`, verifies their SHA-256, builds
the feature cache under `data/maestro/cache/`, and writes 20 s test excerpts to
`outputs/demo_samples/`.

`data/` is gitignored. Downloads are idempotent.

## Train

```bash
uv run python train.py --data-dir data/maestro --output-dir outputs
```

Both models train for up to 25 epochs with early stopping on validation note F1.
Checkpoints go to `outputs/baseline_best.pt` and `outputs/dual_best.pt`, with
per-epoch history in `outputs/history_*.json`.

W&B logging is online only when `WANDB_API_KEY` is set; otherwise it runs
offline (or set `WANDB_MODE=disabled`). Logging failures never stop training.

## Evaluate

```bash
uv run python evaluate.py --data-dir data/maestro \
  --baseline outputs/baseline_best.pt --dual outputs/dual_best.pt \
  --output-dir outputs
```

Writes `outputs/metrics.json` plus `ablation_bar_chart.png`,
`qualitative_pianoroll.png`, and `threshold_sweep.png`.

## Demo

```bash
uv run python app.py --checkpoint outputs/dual_best.pt
```

Then open `http://127.0.0.1:7860` and pick an example from
`outputs/demo_samples/`. `demo.ipynb` covers the same path in a notebook.

## Results (held-out test split)

Two test tracks, `mir_eval.transcription` at a 50 ms onset tolerance and
`offset_ratio=0.2`, with thresholds tuned on the validation split:

| Model | Frame F1 | Note onset F1 | Note onset+offset F1 |
|---|---:|---:|---:|
| Baseline CRNN | 0.495 | 0.812 | 0.341 |
| Onsets & Frames | 0.444 | 0.829 | 0.389 |

The dual head improves note-level onset and onset+offset scores; the baseline
edges it out on dense frame F1. With only two test tracks the noise is large, so
this is a scaled-down ablation, not evidence about the architecture in general.
`outputs/metrics.json` holds the per-track numbers and the full threshold sweep.

## Tests

```bash
uv run pytest -q
```

Contract tests cover mel shape/range, target construction, model output shapes,
rising-edge decoding, repeated notes, chunked inference bounds, and empty
predictions. No network or dataset is needed.

## Limits

- The model trains on ~7 minutes of audio and 103 chunks. This is a
  scaled-down ablation, not a state-of-the-art system.
- 16 kHz audio with `n_fft=2048` gives 7.8 Hz bins, so bass fundamentals are not
  resolved; low notes are inferred from harmonics.
- Sustain pedal is ignored, so sustained notes may end early in the labels.
- Expect poor results on arbitrary recordings, phone microphones, or
  non-piano instruments.
