"""Train the baseline and onsets-and-frames CRNNs on cached MAESTRO chunks."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eardrum import constants as C  # noqa: E402
from eardrum.dataset import MaestroChunks, load_audio, split_tracks  # noqa: E402
from eardrum.decode import decode_notes, predict_probabilities  # noqa: E402
from eardrum.metrics import frame_counts, micro_note_scores, prf  # noqa: E402
from eardrum.model import build_model, count_parameters  # noqa: E402

WANDB_PROJECT = "audio-to-midi-transcription"
DEFAULT_EPOCHS = 60
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12
FRAME_THRESH = 0.15
ONSET_THRESH = 0.25
FRAME_POS_WEIGHT = 4.0
ONSET_POS_WEIGHT = 8.0
ONSET_LOSS_WEIGHT = 1.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_wandb(name: str, config: dict):
    """Online only with an API key; otherwise offline. Never fail training."""
    mode = os.environ.get("WANDB_MODE")
    if mode == "disabled":
        return None
    if mode is None:
        mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
        os.environ["WANDB_MODE"] = mode
    try:
        import wandb

        return wandb.init(project=WANDB_PROJECT, name=name, config=config)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[wandb] disabled ({exc})")
        return None


def cached_metrics(model, loader, device) -> dict[str, float]:
    """Frame/onset F1 over the cached validation chunks."""
    model.eval()
    frame_tp = frame_fp = frame_fn = 0
    onset_tp = onset_fp = onset_fn = 0
    has_onset = False
    with torch.no_grad():
        for mel, frame, onset in loader:
            outputs = model(mel.unsqueeze(1).to(device))
            pred = (torch.sigmoid(outputs["frame"]) >= FRAME_THRESH).cpu().numpy()
            tp, fp, fn = frame_counts(pred, frame.numpy())
            frame_tp, frame_fp, frame_fn = frame_tp + tp, frame_fp + fp, frame_fn + fn
            if "onset" in outputs:
                has_onset = True
                opred = (
                    torch.sigmoid(outputs["onset"]) >= ONSET_THRESH
                ).cpu().numpy()
                tp, fp, fn = frame_counts(opred, onset.numpy())
                onset_tp, onset_fp, onset_fn = onset_tp + tp, onset_fp + fp, onset_fn + fn
    result = {"frame_f1": prf(frame_tp, frame_fp, frame_fn)["f1"]}
    result["onset_head_f1"] = (
        prf(onset_tp, onset_fp, onset_fn)["f1"] if has_onset else None
    )
    return result


TONSET_SWEEP = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
TFRAME_SWEEP = (0.10, 0.15, 0.20, 0.25, 0.30)


def _track_probabilities(model, tracks, device) -> list[tuple]:
    cached = []
    for track in tracks:
        audio = load_audio(track["audio"])
        probs = predict_probabilities(model, audio, device=device)
        cached.append((probs, track["notes"], len(audio) / C.SAMPLE_RATE))
    return cached


def _decode_rows(cached, onset_thresh: float, frame_thresh: float) -> list[tuple]:
    rows = []
    for probs, notes, duration in cached:
        estimated = decode_notes(
            probs["frame"],
            probs["onset"],
            onset_thresh=onset_thresh,
            frame_thresh=frame_thresh,
        )
        rows.append((notes, [(p, s, e) for p, s, e, _ in estimated], duration))
    return rows


def validation_metrics(model, val_loader, val_tracks, device) -> dict[str, float]:
    metrics = cached_metrics(model, val_loader, device)
    cached = _track_probabilities(model, val_tracks, device)
    metrics.update(micro_note_scores(_decode_rows(cached, ONSET_THRESH, FRAME_THRESH)))

    # Fixed thresholds can report 0 for many epochs and trigger a premature
    # stop, so the checkpoint criterion is the best onset threshold on the
    # validation split. The final operating point is tuned after training.
    best_score, best_onset = -1.0, ONSET_THRESH
    for onset_thresh in TONSET_SWEEP:
        score = micro_note_scores(_decode_rows(cached, onset_thresh, FRAME_THRESH))[
            "note_f1"
        ]
        if score > best_score:
            best_score, best_onset = score, onset_thresh
    metrics["tuned_note_f1"] = best_score
    metrics["best_onset_thresh"] = best_onset
    return metrics


def tune_thresholds(checkpoint_path: Path, val_tracks, device) -> tuple[float, float, float]:
    """Pick the validation operating point and store it in the checkpoint."""
    from eardrum.decode import load_model

    model, _ = load_model(checkpoint_path, device)
    cached = _track_probabilities(model, val_tracks, device)
    best = (-1.0, ONSET_THRESH, FRAME_THRESH)
    for onset_thresh in TONSET_SWEEP:
        for frame_thresh in TFRAME_SWEEP:
            scores = micro_note_scores(
                _decode_rows(cached, onset_thresh, frame_thresh)
            )
            if scores["note_f1_offset"] > best[0]:
                best = (scores["note_f1_offset"], onset_thresh, frame_thresh)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    config = dict(checkpoint.get("config", {}))
    config.update(
        {"onset_thresh": best[1], "frame_thresh": best[2], "val_note_f1_offset": best[0]}
    )
    checkpoint["config"] = config
    torch.save(checkpoint, checkpoint_path)
    return best[1], best[2], best[0]


def roll_image(sample, model, device, title: str):
    """Mel + ground truth + prediction for one validation chunk."""
    mel, frame, _ = sample
    model.eval()
    with torch.no_grad():
        outputs = model(mel[None, None].to(device))
    pred = (torch.sigmoid(outputs["frame"])[0] >= FRAME_THRESH).cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].imshow(mel.cpu().numpy(), aspect="auto", origin="lower", cmap="magma")
    axes[0].set_ylabel("mel bin")
    axes[1].imshow(frame.cpu().numpy(), aspect="auto", origin="lower", cmap="gray_r")
    axes[1].set_ylabel("ground truth")
    axes[2].imshow(pred, aspect="auto", origin="lower", cmap="gray_r")
    axes[2].set_ylabel("predicted")
    axes[2].set_xlabel("frame (31.25 Hz)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def train_model(
    name: str,
    args,
    device: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_dataset: MaestroChunks,
    val_tracks: list[dict],
) -> dict:
    model = build_model(name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    frame_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(FRAME_POS_WEIGHT, device=device)
    )
    onset_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(ONSET_POS_WEIGHT, device=device)
    )

    run = init_wandb(
        name,
        {
            "model": name,
            "epochs": args.epochs,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "parameters": count_parameters(model),
            "train_chunks": len(train_loader.dataset),
        },
    )

    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    checkpoint_path = Path(args.output_dir) / f"{name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "frame": 0.0, "onset": 0.0}
        batches = 0
        for mel, frame, onset in train_loader:
            mel = mel.unsqueeze(1).to(device)
            frame = frame.to(device)
            onset = onset.to(device)
            outputs = model(mel)
            frame_loss = frame_loss_fn(outputs["frame"], frame)
            loss = frame_loss
            onset_loss = torch.zeros((), device=device)
            if "onset" in outputs:
                onset_loss = onset_loss_fn(outputs["onset"], onset)
                loss = loss + ONSET_LOSS_WEIGHT * onset_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals["loss"] += loss.detach().item()
            totals["frame"] += frame_loss.detach().item()
            totals["onset"] += onset_loss.detach().item()
            batches += 1
        scheduler.step()
        batches = max(batches, 1)

        metrics = validation_metrics(model, val_loader, val_tracks, device)
        record = {
            "epoch": epoch,
            "train_loss": totals["loss"] / batches,
            "train_frame_loss": totals["frame"] / batches,
            "train_onset_loss": totals["onset"] / batches,
            **{
                key: (float(value) if value is not None else None)
                for key, value in metrics.items()
            },
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        note_f1 = metrics["tuned_note_f1"]
        print(
            f"[{name}] {epoch:02d} loss={record['train_loss']:.4f} "
            f"frame_f1={metrics['frame_f1']:.3f} note_f1={note_f1:.3f} "
            f"(onset_thresh={metrics['best_onset_thresh']:.2f})"
        )

        if run is not None:
            run.log(
                {
                    "train/loss": record["train_loss"],
                    "train/frame_loss": record["train_frame_loss"],
                    "train/onset_loss": record["train_onset_loss"],
                    "val/frame_f1": metrics["frame_f1"],
                    **(
                        {"val/onset_f1": metrics["onset_head_f1"]}
                        if metrics["onset_head_f1"] is not None
                        else {}
                    ),
                    "val/note_f1": note_f1,
                    "val/note_onset_f1": metrics["onset_f1"],
                    "val/note_f1_fixed": metrics["note_f1"],
                    "lr": record["lr"],
                }
            )
            if epoch % 5 == 0:
                import wandb

                image = roll_image(val_dataset[0], model, device, f"{name} epoch {epoch}")
                run.log({"rolls": wandb.Image(image, caption=f"{name} epoch {epoch}")})

        if note_f1 > best_f1:
            best_f1 = note_f1
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": name,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_note_f1": note_f1,
                    "config": {
                        "onset_thresh": metrics["best_onset_thresh"],
                        "frame_thresh": FRAME_THRESH,
                        "epochs": args.epochs,
                    },
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"[{name}] early stop at epoch {epoch} (best {best_epoch})")
                break

    onset_thresh, frame_thresh, val_offset_f1 = tune_thresholds(
        checkpoint_path, val_tracks, device
    )
    print(
        f"[{name}] tuned thresholds onset={onset_thresh:.2f} frame={frame_thresh:.2f} "
        f"(val onset+offset F1 {val_offset_f1:.3f})"
    )

    if run is not None:
        run.finish()

    history_path = Path(args.output_dir) / f"history_{name}.json"
    history_path.write_text(json.dumps(history, indent=1) + "\n")
    print(f"[{name}] best val note F1 {best_f1:.3f} at epoch {best_epoch}")
    return {"best_f1": best_f1, "best_epoch": best_epoch}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/maestro")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["baseline", "dual"],
        choices=["baseline", "dual"],
    )
    args = parser.parse_args(argv)

    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = MaestroChunks(args.data_dir, "train")
    val_dataset = MaestroChunks(args.data_dir, "validation")
    val_tracks = split_tracks(args.data_dir, "validation")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(
        f"device={device} train_chunks={len(train_dataset)} "
        f"val_chunks={len(val_dataset)}"
    )

    summary = {}
    for name in args.models:
        summary[name] = train_model(
            name, args, device, train_loader, val_loader, val_dataset, val_tracks
        )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
