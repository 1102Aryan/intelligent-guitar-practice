from pathlib import Path
from scipy.fftpack import fft
import librosa
import numpy as np
import noisereduce as nr
from scipy.signal import get_window
from basic_pitch.inference import predict_and_save
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH


def pitch_detection(file_path):
    """
    Performs Pitch Detection on the audio file using Spotify's pre-trained model.
    
    Parameters:
        file_path: the path of the audio file.
    
    Returns:
        tuple:
            - model_output: Raw output from the pitch detection.
            - midi_data: MIDI representation of the pitch.
            - note_events: Note events extracted from the MIDI data.
    """
    model_output, midi_data, note_events = predict(
        file_path,
        model_or_model_path=ICASSP_2022_MODEL_PATH)
    return model_output, midi_data, note_events
    
def save_midi(file_path):
    """
    Similar to pitch_detection. This method saves the model to the "outputs" folder.
    
    Parameters:
        file_path: the path of the audio file.
    """
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_output, midi_data, note_events = predict_and_save(
        [str(file_path)],
        save_midi=True,
        output_directory=output_dir,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False)
    print("Complete! Saved midi file to outputs")

def midi_to_note(note_events):
    """
    Extracts information such as onset time, offset time, midi pitch, velocity, confidence from the note events.
    Prints the information in a human-readable format.
    """
    for note in note_events:
        onset_time, offset_time, pitch_midi, velocity, confidence_array = note
        pitch_name = librosa.midi_to_note(int(pitch_midi))
        print(f"Note: {pitch_name}, Start: {onset_time:.2f}s, "
              f"End: {offset_time:.2f}s, Confidence: {velocity:.2f}")

def list_notes(note_events):
    notes = []
    for note in note_events:
        onset_time, offset_time, pitch_midi, velocity, confidence_array = note
        notes.append(pitch_midi)
    return notes

def all_validation(note_events, audio_signal, sample_rate):
    for note in note_events:
        onset_time, offset_time, pitch_midi, velocity, confidence_array = note
        pitch_name = librosa.midi_to_note(int(pitch_midi))
        result = validate_pitch_detection(audio_signal, sample_rate, onset_time, offset_time, pitch_name)
        print(result)
        
def pre_processing(y, sample_rate):
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Reduces noise in audio
    y_clean = nr.reduce_noise(y, sample_rate)
    
        
def validate_pitch_detection(audio_signal, sample_rate, start_time, end_time, note_name):
    """
    Checks if pitch detection is accurate using FFT, if not measure how far off.
    Not accurate enough. Need to denoise 
    
    Parameters:
        audio_signal: signal extracted from the audio using librosa
        sample_rate: sample rate
        start_time: onset time
        end_time: offset time
        note_name: note name
    """
    detected_freq = librosa.note_to_hz(note_name)
    start = int(start_time * sample_rate)
    end = int(end_time * sample_rate)
    segment = audio_signal[start:end]
    window = get_window('hann', len(segment))
    segment = segment * window
    N = len(segment)
    yf = np.abs(fft(segment))
    xf = np.linspace(0, sample_rate/2, N//2)
    idx = np.argmax(yf[:N//2])
    fft_peak_freq = xf[idx]
    error_hz = abs(fft_peak_freq - detected_freq)
    error_cents = 1200 * np.log2(fft_peak_freq / detected_freq) if fft_peak_freq > 0 else None
    ratio = fft_peak_freq / detected_freq
    print(ratio)
  
    return {
        "note": note_name,
        "detected_freq": detected_freq,
        "fft_peak_freq": fft_peak_freq,
        "error_hz": round(error_hz, 2),
        "error_cents": round(error_cents, 2)}
    
    
    
    