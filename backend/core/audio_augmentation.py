"""
Audio augmentation using Spotify Pedalboard.
Generates diverse guitar tone variations for training data augmentation.
"""
import numpy as np
import random
from pedalboard import (
    Pedalboard, Compressor, Gain, Reverb, Chorus,
    Distortion, HighpassFilter, LowShelfFilter, PeakFilter, NoiseGate
)


def build_random_pedalboard() -> Pedalboard:
    """
    Builds a randomised guitar effect chain.
    Simulates variations in: amp compression, gain staging, distortion,
    EQ character, room acoustics, and chorus/vibrato.
    """
    effects = []

    # Always: compression to simulate amp/DI character
    effects.append(Compressor(
        threshold_db=random.uniform(-24, -6),
        ratio=random.uniform(2, 8),
        attack_ms=random.uniform(1, 20),
        release_ms=random.uniform(50, 250),
    ))

    # Always: gain staging
    effects.append(Gain(gain_db=random.uniform(-6, 6)))

    # Optional: noise gate (50% chance) – cleans up string noise
    if random.random() < 0.5:
        effects.append(NoiseGate(
            threshold_db=random.uniform(-70, -40),
            ratio=random.uniform(2, 10),
            attack_ms=random.uniform(1, 5),
            release_ms=random.uniform(50, 150),
        ))

    # Optional: light-to-medium distortion (50% chance)
    if random.random() < 0.5:
        effects.append(Distortion(drive_db=random.uniform(5, 25)))

    # Optional: low-shelf EQ (60% chance) – simulates different cab voicings
    if random.random() < 0.6:
        effects.append(LowShelfFilter(
            cutoff_frequency_hz=random.uniform(60, 250),
            gain_db=random.uniform(-8, 8),
        ))

    # Optional: high-pass filter (40% chance) – removes mud/hum
    if random.random() < 0.4:
        effects.append(HighpassFilter(
            cutoff_frequency_hz=random.uniform(40, 120),
        ))

    # Optional: peak/mid-frequency boost (40% chance) – presence boost
    if random.random() < 0.4:
        effects.append(PeakFilter(
            cutoff_frequency_hz=random.uniform(800, 4000),
            gain_db=random.uniform(-6, 8),
            q=random.uniform(0.5, 3.0),
        ))

    # Optional: reverb (40% chance) – room / small-space acoustics
    if random.random() < 0.4:
        effects.append(Reverb(
            room_size=random.uniform(0.05, 0.4),
            wet_level=random.uniform(0.05, 0.25),
            dry_level=0.8,
        ))

    # Optional: chorus / vibrato (30% chance)
    if random.random() < 0.3:
        effects.append(Chorus(
            rate_hz=random.uniform(0.3, 4.0),
            depth=random.uniform(0.1, 0.4),
            mix=random.uniform(0.1, 0.3),
        ))

    return Pedalboard(effects)


def augment_audio(audio: np.ndarray, sr: int = 22050, n_augmentations: int = 3) -> list:
    """
    Applies n_augmentations independent random pedalboard chains to an audio slice.

    Args:
        audio: 1-D float32/float64 numpy array (mono).
        sr: Sample rate of the audio.
        n_augmentations: Number of augmented copies to produce.

    Returns:
        List of augmented np.ndarray copies (float32, same length as input).
        The original signal is NOT included – handle that separately.
    """
    augmented = []
    for _ in range(n_augmentations):
        board = build_random_pedalboard()
        # pedalboard expects (channels, samples) float32
        audio_f32 = audio.astype(np.float32)
        processed = board(audio_f32[np.newaxis, :], sample_rate=sr)[0]
        # Normalise to prevent clipping
        peak = np.max(np.abs(processed))
        if peak > 1e-6:
            processed = processed / peak
        augmented.append(processed)
    return augmented
