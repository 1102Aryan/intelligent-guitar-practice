from pathlib import Path
from scipy.fftpack import fft
import librosa
import numpy as np
import noisereduce as nr

from scipy.signal import get_window
from scipy import signal
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

def resample_audio(audio_signal, sample_rate):
    if sample_rate != 22050:
        return librosa.resample(audio_signal, orig_sr=sample_rate, target_sr=22050)
    return audio_signal
        
def reduce_noise(audio_signal, sample_rate):
    # Reduces noise in audio
    return nr.reduce_noise(audio_signal, sample_rate)
    
def harmonics_seperation(audio_signal):
    y_harmonic, y_percussive = librosa.effects.hpss(audio_signal)
    return y_harmonic, y_percussive
    
def high_pass_filter(audio_signal, sample_rate):
    # FIR 
    # settings for FIR length (numtaps) : low = 64, medium = 256, high = 1024
    # cutoff for guitar = 80Hz to 100Hz
    # https://mpastell.com/pweave/_downloads/FIR_design_rst.html
    # num taps need to be an odd number
    num_taps = 1025
    cutoff = 80
    normalised = cutoff / (sample_rate/2)
    fir_coefficient = signal.firwin(num_taps, normalised, window="hann", pass_zero=False)
    # Convolution function
    filtered_audio = signal.lfilter(fir_coefficient, 1.0, audio_signal)
    return filtered_audio

def remove_dc(audio_signal):
    """
    Restores the waveform symmetry to be around zero amplitude
    """
    return audio_signal - np.mean(audio_signal)

def normalise(audio_signal):
    """
    Turn audio signal to overall amplitude to a target level.
    """
    highest_amp = max(abs(audio_signal))
    return audio_signal / highest_amp
    

def pre_process(audio_signal, sample_rate):
    """
    Creates a process pipeline which runs all the pre-processes.
    
    
    Returns:
        A filtered audio signal
    """

    # 1: Resample the audio for basic pitch
    audio_signal = resample_audio(audio_signal, sample_rate)

    # 2: Remove DC offset
    audio_signal = remove_dc(audio_signal)

    # 3: normalise audio
    audio_signal = normalise(audio_signal)

    # 4: Remove noise
    audio_signal = reduce_noise(audio_signal, 22050)

    # 5: High pass filter
    audio_signal = high_pass_filter(audio_signal, 22050)

    return audio_signal, 22050
        
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
    
    
    
    