"""Contract tests: shapes, labels, decoding, and empty-input behaviour."""

import numpy as np
import pytest
import torch

from eardrum import constants as C
from eardrum.dataset import mel_spectrogram, notes_to_rolls
from eardrum.decode import (
    decode_notes,
    notes_to_midi,
    predict_probabilities,
    transcribe_audio,
)
from eardrum.metrics import note_scores
from eardrum.model import build_model


def test_mel_contract():
    rng = np.random.default_rng(0)
    chunk = rng.standard_normal(C.N_SAMPLES).astype(np.float32)
    mel = mel_spectrogram(chunk)
    assert mel.shape == (C.N_MELS, C.N_FRAMES)
    assert mel.dtype == np.float32
    assert np.isfinite(mel).all()
    assert -1.001 <= mel.min() and mel.max() <= 1.001


def test_notes_to_rolls_marks_onset_and_tail():
    frame, onset = notes_to_rolls([(60, 1.0, 1.5)], chunk_start=0.0)
    assert frame.shape == (C.N_PITCHES, C.N_FRAMES)
    key = 60 - C.PITCH_MIN
    assert onset[key].sum() == 2  # onset frame plus the +1 tolerance frame
    assert np.argmax(onset[key]) == round(1.0 * C.FRAME_RATE)
    # A frame inside the note is active; the frame before it is not.
    assert frame[key, 32] == 1.0
    assert frame[key, 0] == 0.0


def test_notes_to_rolls_keeps_note_started_before_chunk():
    frame, onset = notes_to_rolls([(60, -1.0, 0.5)], chunk_start=0.0)
    key = 60 - C.PITCH_MIN
    assert frame[key, 0] == 1.0
    assert frame[key, 15] == 1.0
    assert onset[key].sum() == 0


def test_notes_to_rolls_skips_out_of_range_pitches():
    frame, onset = notes_to_rolls(
        [(20, 0.0, 1.0), (109, 0.0, 1.0), (60, 0.0, 1.0)], chunk_start=0.0
    )
    assert frame.sum() > 0
    # Only the in-range note survives: its row is nonzero, every other row is 0.
    assert np.count_nonzero(frame.sum(axis=1)) == 1
    assert np.count_nonzero(onset.sum(axis=1)) == 1


@pytest.mark.parametrize("name", ["baseline", "dual"])
def test_model_output_shapes(name):
    model = build_model(name)
    mel = torch.randn(2, 1, C.N_MELS, C.N_FRAMES)
    outputs = model(mel)
    assert outputs["frame"].shape == (2, C.N_PITCHES, C.N_FRAMES)
    if name == "dual":
        assert outputs["onset"].shape == (2, C.N_PITCHES, C.N_FRAMES)


def _blank_probs(n_frames=126):
    return np.zeros((C.N_PITCHES, n_frames), dtype=np.float32)


def test_decode_rising_edge_ignores_onset_tail():
    frame = _blank_probs()
    onset = _blank_probs()
    key = 60 - C.PITCH_MIN
    onset[key, 0] = 0.9
    onset[key, 1] = 0.9  # the +1 target frame must not retrigger
    frame[key, :40] = 0.9
    notes = decode_notes(frame, onset, onset_thresh=0.45, frame_thresh=0.4)
    assert len(notes) == 1
    assert notes[0][0] == 60
    assert notes[0][1] == 0.0


def test_decode_repeated_note_retriggers():
    frame = _blank_probs()
    onset = _blank_probs()
    key = 48 - C.PITCH_MIN
    onset[key, 5] = 0.9
    onset[key, 6] = 0.9
    onset[key, 25] = 0.9
    frame[key, 5:50] = 0.9
    notes = decode_notes(frame, onset, onset_thresh=0.45, frame_thresh=0.4)
    assert len(notes) == 2
    assert notes[1][1] > notes[0][1]
    assert notes[0][2] == pytest.approx(25 / C.FRAME_RATE)


def test_decode_drops_short_notes():
    frame = _blank_probs()
    onset = _blank_probs()
    key = 72 - C.PITCH_MIN
    onset[key, 50] = 0.9
    frame[key, 50] = 0.9  # one frame only: 0.032 s < 0.05 s
    assert decode_notes(frame, onset) == []


def test_decode_baseline_derives_onsets_from_frames():
    frame = _blank_probs()
    frame[39, :] = 0.9  # flat, so no positive difference
    assert decode_notes(frame, None) == []


def test_predict_probabilities_empty_and_bounded():
    model = build_model("baseline")
    empty = predict_probabilities(model, np.zeros(0, dtype=np.float32))
    assert empty["frame"].shape == (C.N_PITCHES, 0)
    assert empty["onset"] is None

    audio = np.random.default_rng(1).standard_normal(C.SAMPLE_RATE * 5).astype(np.float32)
    probs = predict_probabilities(model, audio)
    expected_frames = 1 + len(audio) // C.HOP_LENGTH
    assert probs["frame"].shape == (C.N_PITCHES, expected_frames)
    assert 0.0 <= probs["frame"].min() and probs["frame"].max() <= 1.0


def test_transcribe_stays_within_duration():
    model = build_model("dual")
    audio = np.random.default_rng(2).standard_normal(C.SAMPLE_RATE * 5).astype(np.float32)
    midi = transcribe_audio(model, audio)
    duration = len(audio) / C.SAMPLE_RATE
    for instrument in midi.instruments:
        for note in instrument.notes:
            assert 0.0 <= note.start < note.end <= duration + 1e-6


def test_empty_prediction_scores_zero():
    reference = [(60, 0.0, 0.5)]
    scores = note_scores(reference, [])
    assert scores["onset_f1"] == 0.0
    assert scores["note_f1"] == 0.0


def test_empty_midi_is_playable_guard():
    midi = notes_to_midi([], duration=1.0)
    assert len(midi.instruments) == 1
    assert midi.instruments[0].notes == []
