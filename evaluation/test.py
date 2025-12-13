import jams
import soundfile as sf
import os
import mir_eval as mir
import numpy as np
from os import listdir
from os.path import isfile, join
from backend.analysis.note_filters import *
from backend.core.audio_loader import audio_loader
from backend.core.pitch_detector import *
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Global lists to store scores for final average
global_precision = []
global_recall = []
global_f1 = []
total_actual_notes = 0
total_predicted_notes = 0

def load_guitarset(file_path):
    try:
        file = jams.load(file_path, strict=False)
    except:
        return []

    notes = []
    for annotation in file.annotations:
        if annotation.namespace != 'note_midi':
            continue
        for note in annotation:
            if isinstance(note.value, (int, float)):
                midi_val = note.value
            elif isinstance(note.value, dict):
                midi_val = note.value.get('value', 0)
            else:
                midi_val = 0
            
            if midi_val > 0:
                notes.append({'onset': note.time, 'offset': note.time + note.duration, 'midi_value': midi_val})
            
    notes.sort(key=lambda x: x['onset'])
    return notes

def evaluate_mir(actual_notes, predicted_notes):
    """
    Uses Mir_eval to evaluate accuracy of pitch notes detection
    """
    global total_actual_notes, total_predicted_notes
    
    predicted_notes.sort(key=lambda x: x[0])
    actual_notes.sort(key=lambda x: x['onset'])

    total_actual_notes += len(actual_notes)
    total_predicted_notes += len(predicted_notes)

    # Convert to Numpy Arrays and Hz for mir_eval
    ref_intervals = np.array([[note['onset'], note['offset']] for note in actual_notes])
    ref_pitches = np.array([440.0 * (2.0 ** ((note['midi_value'] - 69.0) / 12.0)) for note in actual_notes])
    
    est_intervals = np.array([[float(note[0]), float(note[0] + note[1])] for note in predicted_notes])
    est_pitches = np.array([440.0 * (2.0 ** ((float(note[2]) - 69.0) / 12.0)) for note in predicted_notes])
    
    try:
        scores = mir.transcription.evaluate(
            ref_intervals, ref_pitches, est_intervals, est_pitches, 
            onset_tolerance=0.1,
            pitch_tolerance=50.0,
            offset_ratio=None
        )
    except ValueError:
        return {}

    p = scores.get('Precision', scores.get('Precision_no_offset', 0.0))
    r = scores.get('Recall', scores.get('Recall_no_offset', 0.0))
    f = scores.get('F-measure', scores.get('F-measure_no_offset', 0.0))

    global_precision.append(p)
    global_recall.append(r)
    global_f1.append(f)

    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1:        {f:.4f}")
    
    return scores

def final_result():
    avg_precision = sum(global_precision) / len(global_precision) if global_precision else 0
    avg_recall = sum(global_recall) / len(global_recall) if global_recall else 0
    avg_f1 = sum(global_f1) / len(global_f1) if global_f1 else 0

    print("\n==================================================")
    print("                 FINAL RESULTS                    ")
    print("==================================================")
    print(f"Total Actual Notes:    {total_actual_notes}")
    print(f"Total Predicted Notes: {total_predicted_notes}")
    print("--------------------------------------------------")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall:    {avg_recall:.4f}")
    print(f"Average F1 Score:  {avg_f1:.4f}")
    print("==================================================\n")

def evaluate_process():
    path = r"backend/resources/annotation"
    path = os.path.join(BACKEND_ROOT, path)
    
    audio_path = r"backend/resources/audio_mono-mic"
    audio_path = os.path.join(BACKEND_ROOT, audio_path)

    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return

    jam_files = [f for f in listdir(path) if isfile(join(path, f))]
    
    print("Loading Basic Pitch model...")
    basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)

    for jam_path in jam_files:
        print(f"Processing: {jam_path}")
        jam_file_path = join(path, jam_path)
        
        actual_notes = load_guitarset(jam_file_path)
        if not actual_notes:
            continue

        base_name = jam_path.split('.')[0] 
        audio_filename = base_name + '_mic.wav'
        audio_file_path = join(audio_path, audio_filename)
        
        if not os.path.exists(audio_file_path):
            continue
        
        
        # Pre-processing
        audio_signal, sample_rate = audio_loader(audio_file_path)
        audio_signal, sample_rate = pre_process(audio_signal, sample_rate)
        
        # Converts back into wav file for Basic Pitch
        sf.write("processed.wav", audio_signal, sample_rate)
        file_path = "processed.wav"

        try:
            # Using tuned parameters for GuitarSet to reduce ghost notes
            model_output, midi_data, note_events = predict(
                file_path,
                basic_pitch_model,
                onset_threshold=0.6,
                frame_threshold=0.4,
                minimum_note_length=58.0,
                minimum_frequency=None,
                maximum_frequency=None
            )
        except Exception:
            continue
        
        predicted_notes = filter_process(note_events)
        
        
        if predicted_notes:
            evaluate_mir(actual_notes, predicted_notes)

    final_result()

def main():
    evaluate_process()

if __name__ == '__main__':
    main()