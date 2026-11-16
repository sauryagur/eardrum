# SPEC: Dual-Head Polyphonic Piano Transcription (corrected)

Status: approved for implementation. Replaces the TensorFlow prototype in this repo.
Scope: 1–2 days of engineering, CPU-only, one PyTorch system, one tensor contract.

## Corrections vs the original spec

1. **Data source.** The original `googleapis.commaestro-v3.0.0.json` URL does not resolve. The official MAESTRO page publishes only archives; no per-file WAV URLs are published, and guessed paths return 404. The named HF datasets (`roszcz/…`, `dddmr/maestro-tiny`, `amaai-lab/…`) are not publicly resolvable (401, same as nonexistent). Use the official metadata JSON plus the public per-file mirror `ddPn08/maestro-v3.0.0`, pinned to a commit. `ashraq/esc50` is environmental sound with no MIDI and is removed.
2. **Split.** The original had no validation set while selecting checkpoints on validation F1. Use 6 train / 2 validation / 2 test tracks from MAESTRO's official split.
3. **Ablation parity.** Original gave the baseline 15 epochs and the dual-head 25. Both get the same budget, optimizer, and schedule.
4. **Architecture.** The original `Conv1d(128→192)` treats mel bins as channels, mixing all frequencies per step. Use a small 2D CNN so frequency locality is preserved, then a BiGRU.
5. **Entrypoints.** Original used flat modules plus `requirements.txt` while the repo uses `uv`/`pyproject.toml`. Keep `uv`, add a CPU-only torch index, drop `torchaudio` and `scikit-learn`.
6. **Demo inference.** Original stitched non-overlapping 4 s windows, which cuts notes at boundaries. Use 50% overlap with probability averaging.
7. **Claims.** Two test tracks cannot support SOTA or "industry-standard accuracy" claims. Report a scaled-down ablation with explicit caveats.

## Fixed tensor contract

| Item | Value |
|---|---|
| Sample rate | 16000 Hz, mono, float32 |
| Chunk | 4.0 s = 64000 samples |
| Mel | `n_mels=128`, `n_fft=2048`, `hop_length=512`, `center=True` → `(128, 126)` |
| Frame rate | `16000 / 512 = 31.25` Hz (32 ms) |
| Keys | MIDI 21–108 → 88 classes |
| Frame target | `(88, 126)` binary float32 |
| Onset target | `(88, 126)` binary float32, onset frame and `+1` (clipped) |
| Test tolerance | onset 50 ms, offset ratio 0.2 |

`N_FRAMES=126` is `1 + 64000 // 512`. Assert it; never infer it silently.

**Known limitation:** 16 kHz with `n_fft=2048` gives 7.8 Hz bins, while A0 (27.5 Hz) and A#0 differ by 1.6 Hz. Bass fundamentals are not resolved; the model must use harmonics. Do not claim accurate low-register pitch. `n_fft=4096` would improve this but violates the locked contract.

## Data

Official metadata (verified 200):
`https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.json`

Per-file mirror, pinned revision:
`https://huggingface.co/datasets/ddPn08/maestro-v3.0.0/resolve/67e586b1dcf5925c94a6729fcd5eb7f23d504fac/<audio_filename>`
and the same path with `.wav` replaced by `.midi` (note the extension).

Selection: sort each official split by `duration`, take the shortest 6 train, 2 validation, 2 test. Current set (675 s total, ~123 MB WAV):

| split | duration | audio filename |
|---|---:|---|
| train | 45.17 s | `2011/MIDI-Unprocessed_22_R1_2011_MID--AUDIO_R1-D8_12_Track12_wav.wav` |
| train | 63.38 s | `2008/MIDI-Unprocessed_08_R1_2008_01-05_ORIG_MID--AUDIO_08_R1_2008_wav--5.wav` |
| train | 70.73 s | `2013/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_06_R1_2013_wav--3.wav` |
| train | 75.85 s | `2014/MIDI-UNPROCESSED_21-22_R1_2014_MID--AUDIO_21_R1_2014_wav--2.wav` |
| train | 79.33 s | `2008/MIDI-Unprocessed_11_R2_2008_01-05_ORIG_MID--AUDIO_11_R2_2008_wav--2.wav` |
| train | 79.49 s | `2017/MIDI-Unprocessed_070_PIANO070_MID--AUDIO-split_07-08-17_Piano-e_1-02_wav--2.wav` |
| validation | 45.16 s | `2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3.wav` |
| validation | 53.94 s | `2008/MIDI-Unprocessed_04_R3_2008_01-07_ORIG_MID--AUDIO_04_R3_2008_wav--2.wav` |
| test | 65.95 s | `2008/MIDI-Unprocessed_09_R3_2008_01-07_ORIG_MID--AUDIO_09_R3_2008_wav--2.wav` |
| test | 95.97 s | `2008/MIDI-Unprocessed_10_R2_2008_01-05_ORIG_MID--AUDIO_10_R2_2008_wav--1.wav` |

That yields ~99 train chunks, ~24 validation, ~39 test. Do not quote "~200/~60".

Layout:

```
data/maestro/raw/<audio_filename>       # .wav and .midi, gitignored
data/maestro/split.json                 # {split: [{audio, midi, duration}]}
data/maestro/cache/<split>/<stem>.npz   # mels + targets, gitignored
```

Downloader rules:

- Idempotent: skip existing non-empty files, retry 3× with backoff.
- Compute SHA256 after download and store it in `split.json`; do not claim upstream checksums.
- Write `split.json` deterministically from the selection above.
- Copy a 20 s excerpt of each test track (choose the window with the most active frames) to `outputs/demo_samples/` with a `README.md` containing track names, MAESTRO attribution, and CC BY-NC-SA 4.0.
- Never commit `data/` or the cache.
- If the pinned mirror disappears, `DreamyWanderer/MAESTRO-v3.0-FLAC` also ships per-file audio split by train/test. Prefer fixing the mirror URL to re-encoding a different dataset.

Licensing: MAESTRO is CC BY-NC-SA 4.0. The mirror does not declare a license; treat upstream as authoritative. Course use only; do not redistribute raw audio. Demo excerpts are redistributed under CC BY-NC-SA with attribution.

## Targets

Pure function `notes_to_rolls(notes, chunk_start, n_frames=126)` accepting `(pitch, start_s, end_s)`:

- Skip pitches outside 21–108.
- Frame time for index `i` is `chunk_start + i * 512 / 16000`.
- Frame roll: 1 where `note.start <= t < note.end`, including notes that began before the chunk.
- Onset frame: `round((note.start - chunk_start) * 31.25)`; require `0 <= f < 126`; set `onset[f]` and `onset[min(f+1, 125)]`.
- Pedal (CC64) is ignored in v1. Document that sustained notes may end early in labels.

Mel:

```python
S = librosa.feature.melspectrogram(y=chunk, sr=16000, n_fft=2048,
                                   hop_length=512, n_mels=128,
                                   center=True, power=2.0)
S_db = librosa.power_to_db(S, ref=np.max, top_db=80)
mel = ((S_db + 80.0) / 40.0 - 1.0).astype(np.float32)  # [-1, 1]
```

Pad/crop the time axis to exactly 126 and assert. This normalization is per chunk; use the same function at train, eval, and inference.

## Model

`eardrum/model.py`:

- `AcousticEncoder`: input `(B, 1, 128, 126)`.
  - Conv2d(1→32, 3×3) → BN → GELU
  - Conv2d(32→64, 3×3, stride=(2,1)) → BN → GELU
  - Conv2d(64→96, 3×3, stride=(2,1)) → BN → GELU
  - Flatten frequency: `(B, 96, 32, 126)` → `(B, 126, 3072)` → Linear(3072→192) → GELU
  - Output `(B, 126, 192)`; dropout 0.15.
- `BaselineCRNN`: encoder → BiGRU(192→192) → Linear(192→88) → `frame_logits (B, 88, 126)`.
- `OnsetsAndFramesCRNN`:
  - Onset: encoder → BiGRU(192→192) → Linear(192→88) → `onset_logits (B, 88, 126)`.
  - Frame: concat encoder `(B,126,192)` with `sigmoid(onset_logits).detach().permute(0,2,1)` `(B,126,88)` = 280 → BiGRU(280→192) → Linear(192→88) → `frame_logits (B,88,126)`.
- Make every permute explicit. Assert output shapes in tests.

Decoding, `eardrum/decode.py`:

- Start a note on a **rising edge**: `onset[p,t] >= onset_thresh` and (`t == 0` or `onset[p,t-1] < onset_thresh`). The `+1` target frame must not create a second note.
- If a note is already active, a rising edge closes the old note at `t` and starts a new one (repeated notes).
- Close a note when `frame[p,t] < frame_thresh` or at the end.
- Convert with `t / 31.25`; drop notes shorter than 0.05 s; velocity = mean frame probability scaled to 1–127.
- Baseline has no onset head: derive `onset_probs[t] = max(frame_probs[t] - frame_probs[t-1], 0)`.
- Chunked inference for arbitrary length: 64000-sample windows at frame-aligned offsets (`start` a multiple of 512), hop 31744 (62 frames, ~50% overlap). Ensure a final window covers the end. Map local frame `i` to global frame `start/512 + i`, average probabilities with counts, decode, clamp to audio duration. Use this same function in evaluation, app, and notebook.

## Training

`train.py`:

- Seed `random`, `numpy`, `torch`; `torch.set_num_threads` from env if needed.
- Both models: 60 epochs max, batch 16, AdamW(lr=1e-3, weight_decay=1e-4), CosineAnnealingLR, early stopping patience 12, restore best.
- **Epoch budget (measured).** The original 25 epochs is only ~175 optimizer steps on 103 chunks; validation note F1 stayed at 0. 60 epochs (~420 steps) reaches ~0.45. Do not shorten this without re-measuring.
- **Thresholds are validation-tuned.** Frame rate is ~2% active, so a fixed frame threshold of 0.40 suppresses every note and reports F1 = 0. During training, the checkpoint criterion is the best validation note F1 over onset thresholds 0.15–0.40 at frame threshold 0.15, which prevents a zero metric from triggering a premature stop. After training, the operating point is chosen on the validation split over onset 0.15–0.40 × frame 0.10–0.30 maximizing onset+offset F1, then stored in the checkpoint `config` and used by `evaluate.py` and `app.py`. The test split is never used for tuning.
- Loss:
  - dual: `BCEWithLogitsLoss(pos_weight=4.0)(frame) + 1.5 * BCEWithLogitsLoss(pos_weight=8.0)(onset)`
  - baseline: frame term only.
- Validation each epoch: full-track chunked inference on the 2 validation tracks, `mir_eval` note F1 (no offset) at the best swept onset threshold. Checkpoint on that. This avoids the undefined "val" from the original spec.
- Log to W&B: `train/loss`, `train/onset_loss`, `train/frame_loss`, `val/loss`, `val/frame_f1`, `val/onset_f1`, `val/note_f1`, `lr`; `wandb.Image` every 5 epochs with mel, GT, and predicted rolls for the model being trained.
- W&B mode: `online` only if `WANDB_API_KEY` is set, otherwise `offline`; `WANDB_MODE` overrides. Import failure disables logging without failing training.
- Save `outputs/baseline_best.pt`, `outputs/dual_best.pt`, and `outputs/history_*.json`.

## Evaluation

`evaluate.py`:

- Run both checkpoints on the official test split.
- Note metrics with `mir_eval.transcription.evaluate(..., onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05)`. Use keys `Onset_F-measure`, `F-measure_no_offset`, `F-measure`.
- Per-track metrics plus micro metrics: shift each track's intervals by a cumulative offset with a 1 s gap, concatenate, evaluate once. Empty predictions return 0 (verified), so no artificial 1.0.
- Frame metrics: aggregate TP/FP/FN over all test chunks at `frame_thresh=0.4`, then precision/recall/F1. Frame F1 is secondary because rolls are sparse.
- Threshold sweep: onset thresholds 0.2–0.8 step 0.05, frame threshold fixed at 0.4; record both models. Save to `outputs/metrics.json` and `outputs/threshold_sweep.png`.
- `outputs/ablation_bar_chart.png`: frame F1, note onset F1, note onset+offset F1 for both models.
- `outputs/qualitative_pianoroll.png`: 4 panels (mel, GT roll, baseline roll, dual roll) from one test chunk; labels must reflect actual results, no prewritten "smeared vs sharp" narrative.
- Suppress the `mir_eval` empty-estimate warning.

## Demo

`app.py` (Gradio Blocks) and `demo.ipynb`:

- Load `outputs/metrics.json` for a header; if missing, show "metrics not generated".
- Inputs: audio upload/mic (`type="filepath"`), examples from `outputs/demo_samples/`, onset slider 0.1–0.9 and frame slider 0.1–0.9, both defaulting to the validation-tuned values stored in the checkpoint (the spec's 0.45/0.40 defaults are unreachable for this model).
- Outputs: high-res plot (mel + 88-key piano roll, first 30 s if longer), audio playback, `.mid` download, note count and mean duration.
- Audio playback: `midi.synthesize(fs=16000)` (dependency-free sine synth), converted to `int16` in `[-32768, 32767]` before passing to Gradio. Do not require fluidsynth. If `fluidsynth` is available, it may be used behind a flag.
- Guard no-note output (`synthesize` returns an empty array) and missing checkpoint with a clear error.
- Default `server_name=127.0.0.1`, `share=False`; allow `GRADIO_SERVER_NAME` override.
- Notebook runs top-to-bottom from the repo root against a committed checkpoint and a demo excerpt; no training cell.

## Dependencies and layout

Use `pyproject.toml` + `uv.lock`; do not add `requirements.txt`.

Dependencies: `torch>=2.6`, `numpy`, `scipy`, `librosa`, `soundfile`, `pretty_midi`, `mir-eval`, `matplotlib`, `gradio`, `wandb`, `requests`. Remove `tensorflow`, `torchaudio`, `scikit-learn`. Add `[dependency-groups] dev = ["pytest>=8"]`.

CPU torch index:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

Layout:

```
eardrum/
  pyproject.toml
  uv.lock
  eardrum/
    __init__.py
    constants.py
    dataset.py
    model.py
    decode.py
  train.py
  evaluate.py
  app.py
  test_contracts.py
  demo.ipynb
  outputs/
  data/                 # gitignored
```

Delete the TF `eardrum/inference.py`, `eardrum/cli.py`, old `model.py`/`constants.py` contents, `test_eardrum.py`, and the console script. Do not keep both systems.

## Repo hygiene and attribution

- Add `.gitignore`: `.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `wandb/`, `data/`, `outputs/tmp/`.
- Remove tracked `__pycache__/` and `eardrum.egg-info/` (`git rm -r --cached`).
- Copy `LICENSE` and `NOTICE` from `basic-pitch/`; the project remains Apache-2.0 and still adapts Basic Pitch's design. Cite the ICASSP paper.
- Update `README.md` with install, download, train, evaluate, and app commands; state the toy scale, MAESTRO license, and known limits.
- Commit `outputs/dual_best.pt`, `outputs/baseline_best.pt`, `metrics.json`, the three figures, and `demo_samples/` if the total is under ~20 MB; otherwise commit only `dual_best.pt` and the figures. Never commit `data/` or `wandb/`.

## Tests and acceptance

`test_contracts.py` (no network, <60 s):

- Mel shape/dtype/range for 64000 samples; assert 126 frames.
- `notes_to_rolls`: onset marking, note spanning chunk start, out-of-range pitch skipped.
- Both models: forward shapes `(B,88,126)`; dual frame branch accepts onset conditioning.
- Decode: synthetic probabilities produce expected pitch/times; minimum length filter; repeated-note retrigger.
- Chunked transcribe on synthetic audio: output stays within duration; empty prediction handled.
- `mir_eval` empty prediction returns 0.

Acceptance:

```bash
uv sync
uv run python -m eardrum.dataset --download
uv run python train.py --data-dir data/maestro --output-dir outputs
uv run python evaluate.py --data-dir data/maestro --dual outputs/dual_best.pt \
  --baseline outputs/baseline_best.pt --output-dir outputs
uv run python app.py --checkpoint outputs/dual_best.pt
uv run pytest -q
```

Fresh clone: `uv sync` + `uv run pytest` must pass without data; `outputs/` checkpoints and figures are committed so the notebook and app run after clone.

## Timeline (1–2 days)

| Block | Work |
|---|---|
| 1 (half day) | Downloader, cache, target builder, mel, contract tests |
| 2 (half day) | Models, decode, training loop, 1-epoch smoke run |
| 3 (half day) | Full 60-epoch runs, evaluation, figures, metrics.json |
| 4 (half day) | Gradio app, notebook, README/licensing, cleanup |

## Out of scope

- SOTA accuracy, full MAESTRO, velocity/pedal modeling, pitch bends, multi-instrument, real-time, hyperparameter search.
- Any claim that 7 minutes of audio generalizes to arbitrary recordings.
