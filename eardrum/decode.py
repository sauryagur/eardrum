"""Probability inference and onset-conditioned note decoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import torch

from eardrum import constants as C
from eardrum.dataset import load_audio, mel_spectrogram
from eardrum.model import build_model

INFERENCE_BATCH = 8


def predict_probabilities(
    model: torch.nn.Module, audio: np.ndarray, device: str = "cpu"
) -> dict[str, np.ndarray]:
    """Chunked 50%-overlap inference over arbitrary-length audio.

    Returns ``frame`` (88, T) and, for the dual model, ``onset`` (88, T) on the
    global frame grid at 31.25 Hz.
    """
    audio = np.asarray(audio, dtype=np.float32)
    total_frames = 1 + len(audio) // C.HOP_LENGTH
    if len(audio) == 0:
        return {"frame": np.zeros((C.N_PITCHES, 0)), "onset": None}

    last_start = max(total_frames - C.N_FRAMES, 0)
    starts = list(range(0, last_start, C.CHUNK_HOP_FRAMES)) + [last_start]

    windows = []
    for start_frame in starts:
        start_sample = start_frame * C.HOP_LENGTH
        window = audio[start_sample : start_sample + C.N_SAMPLES]
        if len(window) < C.N_SAMPLES:
            window = np.pad(window, (0, C.N_SAMPLES - len(window)))
        windows.append(mel_spectrogram(window))

    frame_sum = np.zeros((C.N_PITCHES, total_frames), dtype=np.float64)
    onset_sum = np.zeros_like(frame_sum)
    counts = np.zeros(total_frames, dtype=np.float64)
    has_onset = False

    model.eval()
    with torch.no_grad():
        for i in range(0, len(windows), INFERENCE_BATCH):
            batch = torch.from_numpy(np.stack(windows[i : i + INFERENCE_BATCH]))
            outputs = model(batch[:, None].to(device))
            frame = torch.sigmoid(outputs["frame"]).cpu().numpy()
            onset = (
                torch.sigmoid(outputs["onset"]).cpu().numpy()
                if "onset" in outputs
                else None
            )
            if onset is not None:
                has_onset = True
            for j, start_frame in enumerate(starts[i : i + INFERENCE_BATCH]):
                valid = min(C.N_FRAMES, total_frames - start_frame)
                sl = slice(start_frame, start_frame + valid)
                frame_sum[:, sl] += frame[j, :, :valid]
                if onset is not None:
                    onset_sum[:, sl] += onset[j, :, :valid]
                counts[sl] += 1

    counts = np.maximum(counts, 1.0)
    frame_probs = (frame_sum / counts).astype(np.float32)
    onset_probs = (onset_sum / counts).astype(np.float32) if has_onset else None
    return {"frame": frame_probs, "onset": onset_probs}


def _frame_difference_onsets(frame_probs: np.ndarray) -> np.ndarray:
    """Baseline has no onset head; use positive frame-probability onsets."""
    diff = np.diff(frame_probs, axis=1, prepend=frame_probs[:, :1])
    return np.clip(diff, 0.0, None)


def decode_notes(
    frame_probs: np.ndarray,
    onset_probs: np.ndarray | None = None,
    onset_thresh: float = 0.45,
    frame_thresh: float = 0.40,
    min_duration: float = C.MIN_NOTE_SECONDS,
) -> list[tuple[int, float, float, int]]:
    """(pitch, start_s, end_s, velocity) notes.

    Onsets are rising edges so the ``+1`` target frame cannot retrigger a note.
    A rising edge while a note is held starts a repeated note.
    """
    if onset_probs is None:
        onset_probs = _frame_difference_onsets(frame_probs)

    notes: list[tuple[int, float, float, int]] = []
    n_frames = frame_probs.shape[1]
    for key in range(C.N_PITCHES):
        pitch = C.PITCH_MIN + key
        active = False
        start = 0
        prob_sum = 0.0
        for t in range(n_frames):
            frame_prob = frame_probs[key, t]
            rising = onset_probs[key, t] >= onset_thresh and (
                t == 0 or onset_probs[key, t - 1] < onset_thresh
            )
            if active and frame_prob < frame_thresh:
                _emit(notes, pitch, start, t, prob_sum, min_duration)
                active, prob_sum = False, 0.0
            if rising:
                if active:
                    _emit(notes, pitch, start, t, prob_sum, min_duration)
                active, start, prob_sum = True, t, 0.0
            if active:
                prob_sum += frame_prob
        if active:
            _emit(notes, pitch, start, n_frames, prob_sum, min_duration)
    notes.sort(key=lambda n: (n[1], n[0]))
    return notes


def _emit(notes, pitch, start, end, prob_sum, min_duration) -> None:
    duration = (end - start) / C.FRAME_RATE
    if duration < min_duration or end <= start:
        return
    velocity = int(round(1 + 126 * prob_sum / (end - start)))
    notes.append((pitch, start / C.FRAME_RATE, end / C.FRAME_RATE, min(max(velocity, 1), 127)))


def notes_to_midi(
    notes: list[tuple[int, float, float, int]], duration: float | None = None
) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for pitch, start, end, velocity in notes:
        if duration is not None:
            start = min(start, duration)
            end = min(end, duration)
        if end <= start:
            continue
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(velocity), pitch=int(pitch), start=float(start), end=float(end)
            )
        )
    midi.instruments.append(instrument)
    return midi


def transcribe_audio(
    model: torch.nn.Module,
    audio: np.ndarray,
    onset_thresh: float = 0.45,
    frame_thresh: float = 0.40,
    device: str = "cpu",
) -> pretty_midi.PrettyMIDI:
    probs = predict_probabilities(model, audio, device=device)
    notes = decode_notes(
        probs["frame"], probs["onset"], onset_thresh=onset_thresh, frame_thresh=frame_thresh
    )
    return notes_to_midi(notes, duration=len(audio) / C.SAMPLE_RATE)


def transcribe_file(
    audio_path: str | Path,
    model: torch.nn.Module,
    onset_thresh: float = 0.45,
    frame_thresh: float = 0.40,
    device: str = "cpu",
) -> pretty_midi.PrettyMIDI:
    return transcribe_audio(
        model,
        load_audio(audio_path),
        onset_thresh=onset_thresh,
        frame_thresh=frame_thresh,
        device=device,
    )


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Restore a checkpoint saved by ``train.py``."""
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model = build_model(checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint.get("config", {})
