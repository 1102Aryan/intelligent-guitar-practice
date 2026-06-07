# https://github.com/spotify/basic-pitch/blob/main/basic_pitch/nn.py
import librosa
import numpy as np


def compute_harmonic_cqt(audio_slice, sr=22050, n_harmonics=6, bins_per_octave=36, include_onset=False):
    """
    Computes a harmonically-stacked CQT, using Librosa.

    Args:
        audio_slice:    1-D mono audio array.
        sr:             Sample rate.
        n_harmonics:    Number of harmonic channels to stack (6 for GOAT v1, 8 for v2).
        bins_per_octave: CQT resolution.
        include_onset:  If True, append a normalised onset-strength channel as the
                        final channel.  The onset envelope is broadcast across the
                        frequency axis so the CNN can cross-correlate it with pitch.

    Returns:
        np.array of shape (n_harmonics [+ 1], n_bins, time_frames)
    """
    n_bins = 4 * bins_per_octave  # 144 bins covering ~4 octaves

    cqt = librosa.cqt(
        y=audio_slice,
        sr=sr,
        hop_length=512,
        n_bins=n_bins + 12 * n_harmonics,  # extra bins for harmonic shifting
        bins_per_octave=bins_per_octave,
        fmin=librosa.note_to_hz('C2')
    )
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    # Harmonic stacking: shift by bins_per_octave * log2(harmonic_number)
    channels = []
    for h in range(1, n_harmonics + 1):
        shift = int(round(bins_per_octave * np.log2(h)))
        shifted = cqt_db[shift:shift + n_bins, :]
        if shifted.shape[0] < n_bins:
            pad = n_bins - shifted.shape[0]
            shifted = np.pad(shifted, ((0, pad), (0, 0)), mode='constant')
        channels.append(shifted)

    if include_onset:
        # Onset-strength envelope tells the CNN *where* the note attack is
        onset_env = librosa.onset.onset_strength(y=audio_slice, sr=sr, hop_length=512)
        n_time = channels[0].shape[1]
        if len(onset_env) >= n_time:
            onset_env = onset_env[:n_time]
        else:
            onset_env = np.pad(onset_env, (0, n_time - len(onset_env)))
        peak = np.max(onset_env)
        if peak > 1e-6:
            onset_env = onset_env / peak
        # Broadcast across frequency bins: shape (n_bins, time)
        onset_channel = np.tile(onset_env[np.newaxis, :], (n_bins, 1))
        channels.append(onset_channel)

    # Shape: (n_harmonics [+1], n_bins, time)
    return np.stack(channels, axis=0)