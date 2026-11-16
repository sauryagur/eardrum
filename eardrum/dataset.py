"""MAESTRO download, feature/target construction, and chunk caching.

The official MAESTRO page publishes only archives, so audio comes from the
public per-file mirror ``ddPn08/maestro-v3.0.0`` pinned to a revision. Metadata
still comes from Google's official JSON so the split and durations are
authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import requests
import soundfile as sf
from torch.utils.data import Dataset

from eardrum import constants as C

METADATA_URL = (
    "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
    "maestro-v3.0.0.json"
)
MIRROR_REPO = "ddPn08/maestro-v3.0.0"
MIRROR_REVISION = "67e586b1dcf5925c94a6729fcd5eb7f23d504fac"
MIRROR_URL = (
    "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
)
SPLIT_SIZES = {"train": 6, "validation": 2, "test": 2}
DEMO_SECONDS = 20


# --------------------------------------------------------------------------- #
# Feature / target building
# --------------------------------------------------------------------------- #
def load_audio(path: str | Path) -> np.ndarray:
    """Mono float32 at the contract sample rate."""
    audio, _ = librosa.load(str(path), sr=C.SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def log_mel(audio: np.ndarray) -> np.ndarray:
    """(128, T) log-mel in [-1, 1] for any-length audio."""
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=C.SAMPLE_RATE,
        n_fft=C.N_FFT,
        hop_length=C.HOP_LENGTH,
        n_mels=C.N_MELS,
        center=True,
        power=2.0,
    )
    spec_db = librosa.power_to_db(spec, ref=np.max, top_db=80.0)
    return ((spec_db + 80.0) / 40.0 - 1.0).astype(np.float32)


def mel_spectrogram(chunk: np.ndarray) -> np.ndarray:
    """(128, 126) log-mel in [-1, 1] for a 64000-sample chunk."""
    assert chunk.shape[-1] == C.N_SAMPLES, chunk.shape
    mel = log_mel(chunk)
    assert mel.shape == (C.N_MELS, C.N_FRAMES), mel.shape
    return mel


def load_notes(midi_path: str | Path) -> list[tuple[int, float, float]]:
    """(pitch, start_s, end_s) for notes inside the 88-key range."""
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [
        (int(n.pitch), float(n.start), float(n.end))
        for inst in midi.instruments
        for n in inst.notes
        if C.PITCH_MIN <= n.pitch <= C.PITCH_MAX
    ]
    notes.sort(key=lambda n: (n[1], n[0]))
    return notes


def notes_to_rolls(
    notes: list[tuple[int, float, float]], chunk_start: float
) -> tuple[np.ndarray, np.ndarray]:
    """Frame and onset piano rolls for one chunk.

    A note that began before ``chunk_start`` is active in the frame roll but
    contributes no onset, which is what the model sees at inference too.
    """
    frame = np.zeros((C.N_PITCHES, C.N_FRAMES), dtype=np.float32)
    onset = np.zeros((C.N_PITCHES, C.N_FRAMES), dtype=np.float32)
    times = chunk_start + np.arange(C.N_FRAMES) * C.HOP_LENGTH / C.SAMPLE_RATE
    for pitch, start, end in notes:
        if not (C.PITCH_MIN <= pitch <= C.PITCH_MAX):
            continue
        key = pitch - C.PITCH_MIN
        active = (start <= times) & (times < end)
        frame[key, active] = 1.0
        f = int(round((start - chunk_start) * C.FRAME_RATE))
        if 0 <= f < C.N_FRAMES:
            onset[key, f] = 1.0
            onset[key, min(f + 1, C.N_FRAMES - 1)] = 1.0
    return frame, onset


def build_track_cache(
    audio_path: str | Path, midi_path: str | Path
) -> dict[str, np.ndarray]:
    """Non-overlapping 4 s chunks for one track."""
    audio = load_audio(audio_path)
    notes = load_notes(midi_path)
    n_chunks = len(audio) // C.N_SAMPLES
    mels, frames, onsets = [], [], []
    for i in range(n_chunks):
        start = i * C.N_SAMPLES
        chunk = audio[start : start + C.N_SAMPLES]
        frame, onset = notes_to_rolls(notes, start / C.SAMPLE_RATE)
        mels.append(mel_spectrogram(chunk))
        frames.append(frame)
        onsets.append(onset)
    if n_chunks == 0:
        raise ValueError(f"{audio_path} is shorter than one chunk")
    return {
        "mel": np.stack(mels),
        "frame": np.stack(frames),
        "onset": np.stack(onsets),
    }


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return _sha256(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(tmp, "wb") as fh:
                    for block in response.iter_content(1 << 20):
                        fh.write(block)
            tmp.replace(dest)
            return _sha256(dest)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def _mirror_url(rel_path: str) -> str:
    return MIRROR_URL.format(repo=MIRROR_REPO, rev=MIRROR_REVISION, path=rel_path)


def load_metadata(cache_dir: Path) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / "maestro-v3.0.0.json"
    if not local.exists():
        _download(METADATA_URL, local)
    raw = json.loads(local.read_text())
    return [
        {key: raw[key][str(i)] for key in raw}
        for i in range(len(raw["year"]))
    ]


def select_tracks(rows: list[dict]) -> dict[str, list[dict]]:
    """Shortest N tracks per official split, deterministic and reproducible."""
    selected: dict[str, list[dict]] = {}
    for split, count in SPLIT_SIZES.items():
        candidates = sorted(
            (r for r in rows if r["split"] == split), key=lambda r: r["duration"]
        )[:count]
        selected[split] = [
            {
                "audio": r["audio_filename"],
                "midi": r["midi_filename"],
                "duration": float(r["duration"]),
            }
            for r in candidates
        ]
    return selected


def download_maestro(
    data_dir: str | Path = "data/maestro",
    output_dir: str | Path = "outputs",
) -> dict[str, list[dict]]:
    data_dir = Path(data_dir)
    rows = load_metadata(data_dir)
    selection = select_tracks(rows)

    split_doc: dict[str, list[dict]] = {}
    for split, tracks in selection.items():
        records = []
        for track in tracks:
            audio_rel = track["audio"]
            midi_rel = track["midi"]
            audio_dest = data_dir / "raw" / audio_rel
            midi_dest = data_dir / "raw" / midi_rel
            print(f"[download] {split}: {audio_rel}")
            audio_sha = _download(_mirror_url(audio_rel), audio_dest)
            midi_sha = _download(_mirror_url(midi_rel), midi_dest)
            records.append(
                {
                    "audio": audio_rel,
                    "midi": midi_rel,
                    "duration": track["duration"],
                    "audio_sha256": audio_sha,
                    "midi_sha256": midi_sha,
                }
            )
        split_doc[split] = records

    split_path = data_dir / "split.json"
    split_path.write_text(json.dumps(split_doc, indent=1, sort_keys=True) + "\n")
    print(f"[download] wrote {split_path}")
    write_demo_excerpts(data_dir, Path(output_dir))
    return split_doc


def write_demo_excerpts(data_dir: Path, output_dir: Path) -> None:
    """20 s test excerpts, used by the Gradio example gallery."""
    demo_dir = output_dir / "demo_samples"
    demo_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Demo excerpts",
        "",
        f"{DEMO_SECONDS}-second excerpts of two MAESTRO v3.0.0 test tracks,",
        "16 kHz mono WAV.",
        "",
        "Source: MAESTRO v3.0.0, Magenta / Google.",
        "License: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/).",
        "Course use only; not for redistribution beyond the terms above.",
        "",
    ]
    for track in json.loads((data_dir / "split.json").read_text())["test"]:
        audio_path = data_dir / "raw" / track["audio"]
        notes = load_notes(data_dir / "raw" / track["midi"])
        audio = load_audio(audio_path)
        excerpt = _busiest_window(audio, notes)
        stem = Path(track["audio"]).stem
        sf.write(demo_dir / f"{stem}.wav", excerpt, C.SAMPLE_RATE)
        lines.append(f"- `{stem}.wav` — `{track['audio']}`")
    (demo_dir / "README.md").write_text("\n".join(lines) + "\n")


def _busiest_window(audio: np.ndarray, notes: list[tuple[int, float, float]]) -> np.ndarray:
    window = DEMO_SECONDS
    if len(audio) <= window * C.SAMPLE_RATE:
        return audio
    starts = [n[1] for n in notes]
    best_start, best_count = 0.0, -1
    limit = len(audio) / C.SAMPLE_RATE - window
    for candidate in np.arange(0.0, limit + 1e-9, 1.0):
        count = sum(1 for s in starts if candidate <= s < candidate + window)
        if count > best_count:
            best_start, best_count = float(candidate), count
    start_sample = int(best_start * C.SAMPLE_RATE)
    return audio[start_sample : start_sample + window * C.SAMPLE_RATE]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def cache_path(data_dir: Path, split: str, audio_rel: str) -> Path:
    stem = audio_rel.replace("/", "__")
    stem = stem[: -len(Path(stem).suffix)] if Path(stem).suffix else stem
    return data_dir / "cache" / split / f"{stem}.npz"


def build_caches(data_dir: str | Path = "data/maestro") -> None:
    data_dir = Path(data_dir)
    split_doc = json.loads((data_dir / "split.json").read_text())
    for split, tracks in split_doc.items():
        for track in tracks:
            dest = cache_path(data_dir, split, track["audio"])
            if dest.exists():
                continue
            print(f"[cache] {split}: {track['audio']}")
            arrays = build_track_cache(
                data_dir / "raw" / track["audio"], data_dir / "raw" / track["midi"]
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, **arrays)


def split_tracks(data_dir: str | Path, split: str) -> list[dict]:
    """Audio path, ground-truth notes, and duration for one split."""
    data_dir = Path(data_dir)
    split_doc = json.loads((data_dir / "split.json").read_text())
    return [
        {
            "audio": data_dir / "raw" / track["audio"],
            "notes": load_notes(data_dir / "raw" / track["midi"]),
            "duration": track["duration"],
        }
        for track in split_doc[split]
    ]


class MaestroChunks(Dataset):
    """Cached (mel, frame, onset) chunks for one split."""

    def __init__(self, data_dir: str | Path, split: str):
        import torch

        data_dir = Path(data_dir)
        split_doc = json.loads((data_dir / "split.json").read_text())
        mels, frames, onsets = [], [], []
        for track in split_doc[split]:
            payload = np.load(cache_path(data_dir, split, track["audio"]))
            mels.append(payload["mel"])
            frames.append(payload["frame"])
            onsets.append(payload["onset"])
        self.mels = torch.from_numpy(np.concatenate(mels))
        self.frames = torch.from_numpy(np.concatenate(frames))
        self.onsets = torch.from_numpy(np.concatenate(onsets))
        assert self.mels.shape[1:] == (C.N_MELS, C.N_FRAMES)
        assert self.frames.shape[1:] == (C.N_PITCHES, C.N_FRAMES)

    def __len__(self) -> int:
        return len(self.mels)

    def __getitem__(self, index: int):
        return self.mels[index], self.frames[index], self.onsets[index]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/maestro")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--download",
        action="store_true",
        help="download metadata + selected tracks, then build the feature cache",
    )
    args = parser.parse_args(argv)
    if not args.download:
        parser.error("nothing to do: pass --download")
    download_maestro(args.data_dir, args.output_dir)
    build_caches(args.data_dir)


if __name__ == "__main__":
    main()
