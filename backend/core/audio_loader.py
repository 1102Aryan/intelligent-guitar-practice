"""
Loading audio files
"""
import librosa
from pathlib import Path
import numpy as np

def audio_loader(file_path):
    """
    Takes the audio file path and loads up the audio signal and sample rate.
    
    Parameter:
        file_path: takes the path audio resides in
    
    Return:
        y: audio signal
        sr: sample rate 
    """
    # Converts to file path (this can be moved to main)
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Loads the song
    y, sr = librosa.load(file_path)
    
    # Print shape
    print(f'y: {y[:10]}')
    print(f'shape y: {y.shape}')
    
    return y, sr

def get_audio_duration(audio_data, sample_rate):
    duration = librosa.get_duration(y=audio_data, sr=sample_rate)
    print(f"Duration: {duration:.2f} seconds")
    return duration

def save_audio():
    pass
    
    
    
    