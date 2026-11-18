# SPEC — Automated Music Transcription (piano)

This document is the contract for this repository: what the system computes,
which quantities are fixed, where each rule lives, and what is out of scope.
When code and this file disagree, the code and its tests win; update this file.

## 1. Purpose and scope

Turn a solo piano recording into note events and a MIDI file. The system covers
feature extraction, label generation, training, decoding, note/frame evaluation,
error analysis and a small demo. It is a compact research/teaching system, not a
product: no velocity regression, no pedal output, no streaming.

Two public datasets are used:

- **MAESTRO v3.0.0** — training, validation and the main test set. Official
  `train` / `validation` / `test` split from `maestro-v3.0.0.csv`; never re-split.
- **SMD** (Saarland Music Data) — 50 real piano recordings with aligned MIDI,
  used only for evaluation, error analysis and robustness. Never for training
  or threshold tuning.

## 2. Fixed contract

| Quantity | Value | Source of truth |
|---|---|---|
| Audio | mono, 16 000 Hz | `configs/*.yaml` → `AudioConfig` |
| CQT | 88 bins, MIDI 21–108 (A0–C8), 12 bins/octave | `CQTConfig` |
| Frame hop | 256 samples = 16 ms = **62.5 fps** | `CQTConfig.hop_length` |
| Pitch range | inclusive MIDI 21–108, 88 keys | `LabelConfig` |
| Frame/onset grid | `(88, n_frames)`, aligned 1:1 with CQT frames | `amt/data/midi.py` |
| Training window | 256 frames (~4.1 s) | `TrainConfig.segment_frames` |
| Cache dtype | CQT float16, rolls uint8 | `scripts/cache_*_features.py` |
| Note onset tolerance | 50 ms | `EvalConfig` / mir_eval |
| Note offset tolerance | max(20 % of reference duration, 50 ms) | mir_eval |
| Minimum note length | 3 frames (~48 ms) | decoder / CLI `--min-frames` |

`Config.__post_init__` enforces `cqt.fmin_midi == labels.pitch_low` and
`cqt.n_bins == pitch_high - pitch_low + 1`; a violation raises. Do not hardcode
88 or 62.5 outside the config path.

Label length is never recomputed from `duration * fs`. `notes_to_piano_roll`
receives `n_frames=cqt.shape[1]`, because librosa's frame count and a ceil of
duration disagree by one on some files (ADR 0003).

## 3. Features

`amt/features/cqt.py::compute_log_cqt` returns `(n_bins, n_frames)` float32.
It uses `librosa.cqt` on the magnitude, then `amplitude_to_db(magnitude,
ref=1.0)` — log magnitude with an absolute reference, not `ref=np.max`, so a
note's value does not depend on other content in the clip (ADR 0002). Frame `i`
is centred at `i * hop_length / sr`; labels use the same convention.

## 4. Labels

`amt/data/midi.py` is the only MIDI parser (pretty_midi, ADR 0001):

- Tempo changes and ticks are handled by pretty_midi.
- Sustain pedal (CC64 >= 64) extends a note's sounding duration, but never past
  the next onset of the same pitch and never merging re-struck notes.
- Duplicate notes are removed by `dedupe_notes`.
- Output is continuous note times (`Note`), converted to frame/onset rolls for
  training and kept continuous for note-level evaluation.

The same parsing and pedal rule must be used for training and evaluation.

## 5. Data pipeline and cache

`scripts/cache_maestro_features.py --root <maestro-v3.0.0> --out <cache>` writes
`<cache>/<split>/<stem>.npz` with:

```
cqt          float16 (88, T)   log-magnitude CQT
frame_roll   uint8   (88, T)   1 while a (pedal-extended) note sounds
onset_roll   uint8   (88, T)   1 on each note onset frame
ref_onsets   float64 (N,)      continuous reference note times (s)
ref_offsets  float64 (N,)
ref_pitches  int     (N,)
stem         str
```

plus `<split>_manifest.json`. Caching is resume-safe: existing `.npz` files are
skipped. `scripts/cache_smd_features.py` uses the same layout for
`data/smd/wav_22050_mono` (real recordings, never the MIDI-synthesised audio).

Song-level splitting lives in `amt/data/dataset.py`; groups (whole works) are
never split across train/validation/test. Training uses a seeded random subset
(`--max-train-hours`, `--max-val-pieces`); the validation subset seed is
`cfg.split.seed`, independent of the training seed, so every run selects its
checkpoint on the same pieces.

## 6. Model

`amt/models/cnn.py::FrameCNN`, built by `build_model(cfg)`:

1. Conv stack over (frequency, time), 3x3 kernels, no frequency pooling, time
   dilations 1, 2, 4, 8 (±15 frames ≈ ±240 ms).
2. Optional harmonic dilated convolution after the first conv layer (HPPNet):
   bins `f + round(12*log2(k))` for k = 2..8.
3. Fully connected layer over each frame's `(channels × 88)` column.
4. Frame head; optional onset head.
5. Optional residual per-pitch BiGRU (`temporal: bigru`), 32 units per
   direction, weights shared across pitches, zero-initialised output heads, so
   step 0 equals the CNN alone.

Input normalisation (per-bin mean/std from the training set) is stored as model
buffers, so a checkpoint is self-contained. `predict_piece` runs a whole piece
in chunks with `receptive_radius` frames of real context on each side, which is
equivalent to a single pass (within float tolerance for the GRU).

Model variants and reported runs:

| Config | Model | Run |
|---|---|---|
| `configs/default.yaml` | frame head only | run1 |
| `configs/onset.yaml` | + onset head | run2 |
| `configs/gru.yaml` | + per-pitch BiGRU | run3 |
| `configs/hd_gru.yaml` | + harmonic dilated conv (3.8M params) | run4 |
| `configs/smd_dev.yaml` | small SMD split for development | — |

## 7. Training

`scripts/train.py` → `amt/training/loop.py::train`:

- One "epoch" is a fixed number of random-window steps, by default one expected
  pass over the training audio (not a dataset sweep); pieces are long, windows
  are random.
- 256-frame windows, batch 16, Adam at 1e-3 with cosine decay, mixed precision.
- Frame targets: BCE. Onset targets: BCE with positives weighted ×10 and each
  onset widened by ±1 frame (`ONSET_POS_WEIGHT`, `ONSET_DILATE`), because
  onsets are ~0.3 % of cells and an unweighted head collapses to "never fires".
- `last.pt` and `best.pt` are written every epoch. `best.pt` is chosen by
  validation **note onset+offset F1** decoded at fixed 0.5/0.5 thresholds when
  validation pieces carry reference notes, else by pooled frame F1. Threshold
  tuning happens later, on the chosen checkpoint, and never on test.
- Training resumes from `last.pt` unless `--no-resume`; checkpoints carry their
  `config`, epoch and model.

Reported runs use 10 epochs on 962 MAESTRO pieces (~8770 steps) on one T4.

## 8. Decoding

`amt/evaluation/decode.py::decode_notes`:

- **Frame-only models:** a note starts on a 0→1 rising edge of the thresholded
  frame probability and ends when it falls back.
- **Onset-head models:** a note starts at a local maximum of the onset
  probability `>= onset_thr` (plateau ties resolve to the first frame), ends
  when the frame probability drops below `frame_thr`; frame activity without an
  onset peak is ignored.
- Notes shorter than `min_frames` (default 3) are dropped.

Thresholds are decoding choices, tuned per model family on MAESTRO validation by
note+offset F1. Defaults (`amt/inference.py::DEFAULT_THRESHOLDS`):

| Model family | frame_thr | onset_thr |
|---|---|---|
| frame-only CNN (run1) | 0.5 | 0.5 |
| CNN + onset head (run2) | 0.3 | 0.8 |
| + BiGRU (run3, run4) | 0.5 | 0.9 |

Evaluate/test data never informs thresholds.

## 9. Evaluation

`amt/evaluation/metrics.py`:

- Frame-level P/R/F1 over the aligned `(88, T)` binary grids (pooled or
  per-piece as the caller chooses). Empty reference and empty estimate score
  1.0; any other zero denominator scores 0.0.
- Note-level onset and onset+offset P/R/F1 via `mir_eval.transcription`,
  computed on continuous times, matched per pitch (equivalent to mir_eval's
  global matching at 100-cent spacing).

`scripts/evaluate_model.py` optionally tunes thresholds on `--tune-dir`
(validation), then reports on `--dir`, macro-averaged over pieces. Output JSON
contains the macro means, 95 % bootstrap intervals over pieces (2 000
resamples), and `per_piece` scores. `scripts/compare_runs.py` does a paired
bootstrap between two result files on the same pieces. `scripts/analyze_errors.py`
breaks recall down by note length, register, polyphony and sustain pedal.
`scripts/robustness.py` applies noise, reverb, low-pass and detuning.

Every reported number is checked into `results/` as JSON; `scripts/make_figures.py`
rebuilds `figures/` from those files. The test split is used only for final
reporting.

## 10. Inference and demo

`amt/inference.py`:

- `load_checkpoint` restores the model, its `Config` and device from the
  checkpoint (`weights_only=False`).
- Audio is loaded mono at `cfg.audio.sample_rate`; CQT uses the checkpoint's own
  settings and is round-tripped through float16 to match training caches.
- `transcribe_file` returns notes; `notes_to_midi` writes a single piano track
  at velocity from the note (default 100), tempo 120.
- `scripts/transcribe.py` exposes `--offset`, `--duration`, `--frame-thr`,
  `--onset-thr`, `--min-frames`.

`app.py` is a Gradio demo (piano roll, MIDI download, sine preview). Checkpoint
resolution order: `AMT_CKPT` → `models/run4_best.pt` → download
`AMT_CKPT_URL` (default the v1.0 release) into `~/.cache/amt/`. Audio is capped
by `AMT_MAX_SECONDS` (default 600 s).

## 11. Repository contract

```
amt/            importable package (config, data, features, models, training,
                evaluation, inference, baselines, utils)
scripts/        CLI entry points: caching, training, evaluation, analysis,
                figures, transcription, robustness, speed
configs/        one YAML per reported run; Config validates consistency
results/        committed JSON for every reported number
figures/        committed PDF/PNG built from results/
tests/          pytest suite; torch-dependent tests skip if torch is absent
docs/           PROGRESS.md (dated log) and docs/decisions/ (ADRs)
legacy/         original exploratory scripts, reference only, not imported
```

Dependencies: `numpy`, `scipy`, `librosa`, `pretty_midi`, `soundfile`,
`mir_eval`, `PyYAML`; optional `torch` (`train`), `gradio`/`matplotlib`
(`demo`), `matplotlib` (`figures`), `pytest` (`dev`). The MAESTRO cache script
additionally imports pandas, which is not declared in `pyproject.toml`. The
figures script and app import matplotlib.

## 12. Out of scope / known limits

- No velocity or pedal prediction; velocity is constant.
- 12 bins/octave limits bass and high-register resolution.
- Single seed for run3/run4; differences under ~1 F1 point are noise.
- Trained on one piano type (MAESTRO concert recordings); robustness is tested
  with synthetic degradations only.
- Data and checkpoints are never committed. MAESTRO is CC BY-NC-SA 4.0 — cite
  it; do not redistribute audio. SMD has its own terms.
