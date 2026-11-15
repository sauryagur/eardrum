"""Evaluate both checkpoints on the held-out MAESTRO test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eardrum import constants as C  # noqa: E402
from eardrum.dataset import MaestroChunks, load_audio, split_tracks  # noqa: E402
from eardrum.decode import (  # noqa: E402
    decode_notes,
    load_model,
    predict_probabilities,
)
from eardrum.metrics import (  # noqa: E402
    frame_counts,
    micro_note_scores,
    note_scores,
    prf,
)

FRAME_THRESH = 0.40
ONSET_THRESH = 0.45
THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.05), 2)


def frame_metrics(model, dataset: MaestroChunks, device: str) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    tp = fp = fn = 0
    with torch.no_grad():
        for mel, frame, _ in loader:
            outputs = model(mel.unsqueeze(1).to(device))
            pred = (torch.sigmoid(outputs["frame"]) >= FRAME_THRESH).cpu().numpy()
            a, b, c = frame_counts(pred, frame.numpy())
            tp, fp, fn = tp + a, fp + b, fn + c
    return prf(tp, fp, fn)


def evaluate_model(checkpoint: Path, data_dir: Path, device: str) -> dict:
    model, config = load_model(checkpoint, device)
    dataset = MaestroChunks(data_dir, "test")
    tracks = split_tracks(data_dir, "test")

    frame = frame_metrics(model, dataset, device)

    per_track = {}
    sweep_points = []
    cached: dict[str, tuple] = {}
    for track in tracks:
        audio = load_audio(track["audio"])
        probs = predict_probabilities(model, audio, device=device)
        cached[track["audio"].name] = (probs, track["notes"], len(audio) / C.SAMPLE_RATE)

    def decode_at(onset_thresh: float) -> list[tuple]:
        rows = []
        for probs, notes, duration in cached.values():
            estimated = decode_notes(
                probs["frame"],
                probs["onset"],
                onset_thresh=onset_thresh,
                frame_thresh=FRAME_THRESH,
            )
            rows.append(
                (notes, [(p, s, e) for p, s, e, _ in estimated], duration)
            )
        return rows

    default_rows = decode_at(ONSET_THRESH)
    micro = micro_note_scores(default_rows)

    per_track = {}
    for track, (probs, notes, duration), (_, estimated, _) in zip(
        tracks, cached.values(), default_rows
    ):
        scores = note_scores(notes, estimated)
        scores["notes_predicted"] = len(estimated)
        scores["notes_reference"] = len(notes)
        per_track[Path(track["audio"]).stem] = scores

    for threshold in THRESHOLDS:
        rows = decode_at(float(threshold))
        scores = micro_note_scores(rows)
        sweep_points.append(
            {
                "threshold": float(threshold),
                "precision": scores["onset_precision"],
                "recall": scores["onset_recall"],
                "f1": scores["onset_f1"],
            }
        )

    return {
        "checkpoint": str(checkpoint),
        "config": config,
        "frame": frame,
        "note": micro,
        "per_track": per_track,
        "threshold_sweep": sweep_points,
        "frame_thresh": FRAME_THRESH,
        "onset_thresh": ONSET_THRESH,
    }


def plot_ablation(results: dict, output: Path) -> None:
    labels = ["Frame F1", "Note onset F1", "Note onset+offset F1"]

    def values(model):
        entry = results[model]
        return [
            entry["frame"]["f1"],
            entry["note"]["onset_f1"],
            entry["note"]["note_f1_offset"],
        ]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, values("baseline"), width, label="Baseline CRNN", color="#8c8c8c")
    ax.bar(x + width / 2, values("dual"), width, label="Onsets & Frames", color="#1f77b4")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("MAESTRO test split — scaled-down ablation (2 tracks)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (base, dual) in enumerate(zip(values("baseline"), values("dual"))):
        ax.text(i - width / 2, base + 0.01, f"{base:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, dual + 0.01, f"{dual:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_qualitative(
    results: dict, data_dir: Path, device: str, output: Path
) -> None:
    dataset = MaestroChunks(data_dir, "test")
    index = int(dataset.frames.sum(dim=(1, 2)).argmax())
    mel, frame, _ = dataset[index]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].imshow(mel.numpy(), aspect="auto", origin="lower", cmap="magma")
    axes[0].set_ylabel("mel bin")
    axes[0].set_title("Log-mel input")
    axes[1].imshow(frame.numpy(), aspect="auto", origin="lower", cmap="gray_r")
    axes[1].set_ylabel("MIDI")
    axes[1].set_title("Ground truth")

    for axis, name, title in (
        (axes[2], "baseline", "Baseline CRNN"),
        (axes[3], "dual", "Onsets & Frames"),
    ):
        model, _ = load_model(results[name]["checkpoint"], device)
        with torch.no_grad():
            outputs = model(mel[None, None].to(device))
        pred = (torch.sigmoid(outputs["frame"])[0] >= FRAME_THRESH).cpu().numpy()
        axis.imshow(pred, aspect="auto", origin="lower", cmap="gray_r")
        axis.set_ylabel("MIDI")
        axis.set_title(title)
    axes[3].set_xlabel("frame (31.25 Hz, 4 s)")
    fig.suptitle("Qualitative piano roll on one test chunk")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_threshold_sweep(results: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"baseline": "#8c8c8c", "dual": "#1f77b4"}
    names = {"baseline": "Baseline CRNN", "dual": "Onsets & Frames"}

    for name in results:
        sweep = results[name]["threshold_sweep"]
        recall = [p["recall"] for p in sweep]
        precision = [p["precision"] for p in sweep]
        thresholds = [p["threshold"] for p in sweep]
        axes[0].plot(recall, precision, "o-", color=colors[name], label=names[name])
        axes[1].plot(thresholds, [p["f1"] for p in sweep], "o-", color=colors[name], label=names[name])

    axes[0].set_xlabel("onset recall")
    axes[0].set_ylabel("onset precision")
    axes[0].set_title("Precision–recall across onset thresholds")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("onset threshold")
    axes[1].set_ylabel("note onset F1")
    axes[1].set_title("F1 vs onset threshold (frame threshold fixed)")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/maestro")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--baseline", default="outputs/baseline_best.pt")
    parser.add_argument("--dual", default="outputs/dual_best.pt")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {
        "baseline": evaluate_model(Path(args.baseline), data_dir, device),
        "dual": evaluate_model(Path(args.dual), data_dir, device),
    }

    payload = {
        "dataset": "MAESTRO v3.0.0 test split (2 shortest tracks)",
        "caveat": "Scaled-down ablation on ~2.7 minutes of test audio; not SOTA.",
        "tracks": [track["audio"].name for track in split_tracks(data_dir, "test")],
        "models": results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=1) + "\n")

    plot_ablation(results, output_dir / "ablation_bar_chart.png")
    plot_qualitative(results, data_dir, device, output_dir / "qualitative_pianoroll.png")
    plot_threshold_sweep(results, output_dir / "threshold_sweep.png")

    for name, entry in results.items():
        print(
            f"{name:8s} frame_f1={entry['frame']['f1']:.3f} "
            f"onset_f1={entry['note']['onset_f1']:.3f} "
            f"note_f1={entry['note']['note_f1']:.3f} "
            f"note_f1_offset={entry['note']['note_f1_offset']:.3f}"
        )
    print(f"wrote {output_dir / 'metrics.json'} and 3 figures")


if __name__ == "__main__":
    main()
