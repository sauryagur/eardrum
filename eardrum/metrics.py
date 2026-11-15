"""Frame and note metrics shared by training and evaluation."""

from __future__ import annotations

import warnings

import mir_eval
import numpy as np

from eardrum import constants as C


def frame_counts(
    predicted: np.ndarray, target: np.ndarray
) -> tuple[int, int, int]:
    """Aggregate TP/FP/FN for boolean (B, 88, T) arrays."""
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    tp = int(np.logical_and(predicted, target).sum())
    fp = int(np.logical_and(predicted, ~target).sum())
    fn = int(np.logical_and(~predicted, target).sum())
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def intervals_and_pitches(notes) -> tuple[np.ndarray, np.ndarray]:
    intervals = np.array([[start, end] for _, start, end in notes], dtype=float)
    intervals = intervals.reshape(-1, 2)
    pitches = np.array([pitch for pitch, _, _ in notes], dtype=int)
    return intervals, pitches


def note_scores(ref_notes, est_notes) -> dict[str, float]:
    """mir_eval note metrics at the MIREX 50 ms onset tolerance."""
    ref_i, ref_p = intervals_and_pitches(ref_notes)
    est_i, est_p = intervals_and_pitches(est_notes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = mir_eval.transcription.evaluate(
            ref_i,
            ref_p,
            est_i,
            est_p,
            onset_tolerance=C.ONSET_TOLERANCE,
            offset_ratio=C.OFFSET_RATIO,
            offset_min_tolerance=C.OFFSET_MIN_TOLERANCE,
        )
    return {
        "onset_f1": float(result["Onset_F-measure"]),
        "onset_precision": float(result["Onset_Precision"]),
        "onset_recall": float(result["Onset_Recall"]),
        "note_f1": float(result["F-measure_no_offset"]),
        "note_f1_offset": float(result["F-measure"]),
    }


def micro_note_scores(per_track) -> dict[str, float]:
    """One mir_eval pass over tracks shifted apart by a 1 s gap."""
    ref_all, est_all = [], []
    offset = 0.0
    for ref_notes, est_notes, duration in per_track:
        ref_all.extend((p, s + offset, e + offset) for p, s, e in ref_notes)
        est_all.extend((p, s + offset, e + offset) for p, s, e in est_notes)
        offset += duration + 1.0
    return note_scores(ref_all, est_all)
