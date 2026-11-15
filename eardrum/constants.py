"""Fixed tensor contract for the transcription pipeline.

Do not change a value here without updating SPEC.md and every caller: the
dataset cache, the model, the decoder, the evaluator and the demo all assume
these exact shapes.
"""

SAMPLE_RATE = 16000
CHUNK_SECONDS = 4.0
N_SAMPLES = 64000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
N_FRAMES = 126

# center=True gives 1 + n_samples // hop frames; assert instead of inferring.
assert N_FRAMES == 1 + N_SAMPLES // HOP_LENGTH, "N_FRAMES no longer matches hop"

FRAME_RATE = SAMPLE_RATE / HOP_LENGTH  # 31.25 Hz

PITCH_MIN = 21  # A0
PITCH_MAX = 108  # C8
N_PITCHES = PITCH_MAX - PITCH_MIN + 1

# 50%-overlap windows, frame aligned so local frame i maps to global frame
# start_frame + i with no resampling of the time axis.
CHUNK_HOP_FRAMES = 62
CHUNK_HOP_SAMPLES = CHUNK_HOP_FRAMES * HOP_LENGTH  # 31744

ONSET_TOLERANCE = 0.05
OFFSET_RATIO = 0.2
OFFSET_MIN_TOLERANCE = 0.05

MIN_NOTE_SECONDS = 0.05
