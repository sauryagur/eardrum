"""Gradio demo: upload piano audio, get a piano roll, synth playback, and MIDI."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eardrum import constants as C  # noqa: E402
from eardrum.dataset import load_audio, log_mel  # noqa: E402
from eardrum.decode import (  # noqa: E402
    decode_notes,
    load_model,
    notes_to_midi,
    predict_probabilities,
)

MAX_PLOT_SECONDS = 30.0
C_OCTAVES = [(key, C.PITCH_MIN + key) for key in range(3, C.N_PITCHES, 12)]


def header_markdown(metrics_path: Path) -> str:
    base = (
        "# Eardrum — polyphonic piano transcription\n"
        "Dual-head CRNN (onsets & frames) trained on a small MAESTRO subset. "
        "Upload or record piano audio, then adjust the two thresholds."
    )
    if not metrics_path.exists():
        return base + "\n\n_Test metrics not generated (`python evaluate.py`)._"
    try:
        entry = json.loads(metrics_path.read_text())["models"]["dual"]["note"]
    except (KeyError, json.JSONDecodeError):
        return base + "\n\n_Test metrics file is unreadable._"
    return base + (
        "\n\n**Test-set note scores:** "
        f"onset F1 {entry['onset_f1']:.3f} · "
        f"onset+offset F1 {entry['note_f1_offset']:.3f}. "
        "Two held-out tracks — a scaled-down ablation, not SOTA."
    )


def piano_roll_figure(
    audio: np.ndarray, frame_probs: np.ndarray, frame_thresh: float
) -> plt.Figure:
    max_frames = min(frame_probs.shape[1], int(MAX_PLOT_SECONDS * C.FRAME_RATE))
    samples = min(len(audio), int(MAX_PLOT_SECONDS * C.SAMPLE_RATE))
    mel = log_mel(audio[:samples])
    mel_seconds = mel.shape[1] / C.FRAME_RATE
    roll_seconds = max_frames / C.FRAME_RATE
    seconds = min(mel_seconds, roll_seconds, MAX_PLOT_SECONDS)

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1, 1.2]}
    )
    axes[0].imshow(
        mel,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, mel_seconds, 0, C.N_MELS],
    )
    axes[0].set_ylabel("mel bin")
    axes[0].set_title("Log-mel spectrogram")
    axes[0].set_xlim(0, seconds)

    active = (frame_probs[:, :max_frames] >= frame_thresh).astype(float)
    axes[1].imshow(
        active,
        origin="lower",
        aspect="auto",
        cmap="gray_r",
        extent=[0, roll_seconds, -0.5, C.N_PITCHES - 0.5],
    )
    axes[1].set_yticks([key for key, _ in C_OCTAVES])
    axes[1].set_yticklabels([f"C{(pitch - 12) // 12}" for _, pitch in C_OCTAVES])
    axes[1].set_ylabel("pitch")
    axes[1].set_xlabel("seconds")
    axes[1].set_title(f"Transcribed piano roll (frame threshold {frame_thresh:.2f})")
    axes[1].set_xlim(0, seconds)

    fig.tight_layout()
    return fig


def synth_playback(midi) -> tuple[int, np.ndarray] | None:
    if not any(instrument.notes for instrument in midi.instruments):
        return None
    waveform = midi.synthesize(fs=C.SAMPLE_RATE)
    if waveform.size == 0:
        return None
    peak = float(np.max(np.abs(waveform))) or 1.0
    pcm = np.clip(waveform / peak, -1.0, 1.0)
    return C.SAMPLE_RATE, (pcm * 32767).astype(np.int16)


def transcribe(audio_path, onset_thresh, frame_thresh):
    import gradio as gr

    if not audio_path:
        raise gr.Error("Provide an audio file or record from the microphone.")
    try:
        audio = load_audio(audio_path)
    except Exception as exc:
        raise gr.Error(f"Could not read audio: {exc}") from exc
    if len(audio) < C.SAMPLE_RATE // 10:
        raise gr.Error("Audio is too short; provide at least 0.1 s.")

    probs = predict_probabilities(MODEL, audio, device=DEVICE)
    notes = decode_notes(
        probs["frame"],
        probs["onset"],
        onset_thresh=float(onset_thresh),
        frame_thresh=float(frame_thresh),
    )
    midi = notes_to_midi(notes, duration=len(audio) / C.SAMPLE_RATE)

    figure = piano_roll_figure(audio, probs["frame"], float(frame_thresh))
    playback = synth_playback(midi)

    midi_path = tempfile.NamedTemporaryFile(suffix=".mid", delete=False).name
    midi.write(midi_path)

    if notes:
        durations = [end - start for _, start, end, _ in notes]
        summary = (
            f"**{len(notes)} notes** · mean duration {np.mean(durations):.2f} s · "
            f"pitch range {min(n[0] for n in notes)}–{max(n[0] for n in notes)}"
        )
    else:
        summary = (
            "**No notes detected.** Lower the frame or onset threshold, "
            "or use a cleaner piano recording."
        )
    return figure, playback, midi_path, summary


def build_demo(
    metrics_path: Path,
    onset_default: float = 0.45,
    frame_default: float = 0.40,
):
    import gradio as gr

    examples = sorted(str(path) for path in Path("outputs/demo_samples").glob("*.wav"))
    with gr.Blocks(title="Eardrum — piano transcription") as demo:
        gr.Markdown(header_markdown(metrics_path))
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Piano audio",
                )
                onset = gr.Slider(
                    0.1,
                    0.9,
                    value=onset_default,
                    step=0.05,
                    label="Onset sensitivity threshold",
                )
                frame = gr.Slider(
                    0.1,
                    0.9,
                    value=frame_default,
                    step=0.05,
                    label="Frame sustain threshold",
                )
                button = gr.Button("Transcribe", variant="primary")
            with gr.Column():
                plot = gr.Plot(label="Spectrogram and piano roll")
                playback = gr.Audio(label="Synthesized transcription", type="numpy")
                midi_file = gr.File(label="Download MIDI")
                summary = gr.Markdown()
        if examples:
            gr.Examples(examples=examples, inputs=audio)
        button.click(
            transcribe,
            inputs=[audio, onset, frame],
            outputs=[plot, playback, midi_file, summary],
        )
    return demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/dual_best.pt")
    parser.add_argument("--metrics", default="outputs/metrics.json")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)

    global MODEL, DEVICE
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(
            f"checkpoint not found: {checkpoint}. Run train.py first or pass --checkpoint."
        )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL, config = load_model(checkpoint, DEVICE)

    demo = build_demo(
        Path(args.metrics),
        onset_default=float(config.get("onset_thresh", 0.45)),
        frame_default=float(config.get("frame_thresh", 0.40)),
    )
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=args.port,
        share=args.share,
    )


MODEL = None
DEVICE = "cpu"


if __name__ == "__main__":
    main()
