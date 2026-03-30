# https://github.com/spotify/basic-pitch/blob/main/basic_pitch/nn.py
import librosa
import numpy as np


def compute_harmonic_cqt(audio_slice, sr=22050, n_harmonics=6, bins_per_octave=36):
    """
    Computes a harmonically-stacked CQT, using Librosa.
    Parameter:
        n_harmonic, bins, sr
    Returns:
        np.array of shape (n_harmonics, n_bins, time_frames)
    """
    n_bins = 4 * bins_per_octave  # 144 bins covering ~4 octaves
    
    cqt = librosa.cqt(
        y=audio_slice, 
        sr=sr, 
        hop_length=512,
        n_bins=n_bins + 12 * n_harmonics,  # extra bins for shifting
        bins_per_octave=bins_per_octave,
        fmin=librosa.note_to_hz('C2')
    )
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    
    # Harmonic stacking: shift by bins_per_octave * log2(harmonic_number)
    channels = []
    for h in range(1, n_harmonics + 1):
        shift = int(round(bins_per_octave * np.log2(h)))
        # Shifting down - take from higher frequency bins
        shifted = cqt_db[shift:shift + n_bins, :]
        
        # pad if not enough bins
        if shifted.shape[0] < n_bins:
            pad = n_bins - shifted.shape[0]
            shifted = np.pad(shifted, ((0, pad), (0, 0)), mode='constant')
        
        channels.append(shifted)
    
    # return Shape - (n_harmonics, n_bins, time)
    return np.stack(channels, axis=0)