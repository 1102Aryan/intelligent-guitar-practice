import jams
import soundfile as sf
import os
from os import listdir
from os.path import isfile, join
import sys
import re

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.core.pitch_detector import pitch_detection
from backend.analysis.note_filters import filter_process
from backend.models.fretboard_mapper import FretBoardMapper
from backend.tablature.fretboard_mapper import *
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# global counters
all_true_positives = 0
all_false_negatives = 0
all_false_positives = 0
total_actual_notes = 0
total_predicted_notes = 0
total_pitch_matches = 0
total_string_distance = 0
total_fret_distance = 0
# max searches for the file
LIMIT = 5000000  

STANDARD_TUNING = {
    6: 40, # E2
    5: 45, # A2
    4: 50, # D3
    3: 55, # G3
    2: 59, # B3
    1: 64  # E4
}

def normalize_folder_name(name):
    """
    Simple normalisation
    """
    return name.strip().lower()

def load_guitarset(file_path):
    """
    Loads the guitar set extracting metadata annotations
    returns:
        list of onset, offset, midi_value, string and fret from annotations
    """
    try:
        file = jams.load(file_path, strict=False)
    except:
        return []

    notes = []
    # standard tuning
    STRING_MAP = {
        "0": {"open": 40, "string_num": 6}, # Low E
        "1": {"open": 45, "string_num": 5}, # A
        "2": {"open": 50, "string_num": 4}, # D
        "3": {"open": 55, "string_num": 3}, # G
        "4": {"open": 59, "string_num": 2}, # B
        "5": {"open": 64, "string_num": 1}  # High E
    }
    for annotation in file.annotations:
        if annotation.namespace != 'note_midi':
            continue
        string_number = annotation.annotation_metadata.data_source
        # Ensures only string numbers are accepted
        if string_number not in STRING_MAP:
            continue
        
        string_info = STRING_MAP[string_number]
        open_pitch = string_info["open"]
        string_num = string_info["string_num"]
        for note in annotation:
            if note.value <= 0:
                continue
            fret = int(round(note.value - open_pitch))
            if isinstance(note.value, (int, float)):
                midi_val = note.value
            elif isinstance(note.value, dict):
                midi_val = note.value.get('value', 0)
            else:
                midi_val = 0
            
            if 0 <= fret <= 24:
                notes.append({'onset': note.time,
                              'offset': note.time + note.duration,
                              'midi_value': midi_val,
                              'string': string_num,
                              'fret': fret})
            
    notes.sort(key=lambda x: x['onset'])
    return notes

def fret_string_match(actual_note, predicted_note):
    """
    If the difference is 0 then the fret or string are accurately predicted. 
    """
    diff_string = abs(actual_note['string'] - predicted_note['string'])
    diff_fret = abs(actual_note['fret'] - predicted_note['fret'])
    return diff_string, diff_fret

def get_cnn_predicted_notes(audio_path, mapper):
    """
    Use fretboard CNN to predict the string and fret based on the basic pitch results.
    
    returns:
        predicted list of onset, offset, string, fret
    """
    if not os.path.exists(audio_path):
        return []

    try:
        _, _, note_events = pitch_detection(audio_path)
    except Exception:
        return []
    
    if not note_events:
        return []
    
    # filtered_notes = filter_process(note_events)
    filtered_notes = note_events
    if not filtered_notes:
        return []

    
    midi_list = [note[2] for note in filtered_notes]
    mapped = mapper.map_notes(midi_list)

    predicted_notes = []
    for note_event, position in zip(filtered_notes, mapped):
        predicted_notes.append({
            'onset': note_event[0],
            'offset': note_event[1],
            'string': position[0],
            'fret': position[1]
        })
    return predicted_notes

def format_to_fretboard(note_events, mapper):
    if not note_events:
        return[]
    
    midi_list = [note[2] for note in note_events]
    mapped = mapper.map_notes(midi_list)
    predicted_notes = []
    for note_event, position in zip(note_events, mapped):
        predicted_notes.append({
            'onset': note_event[0],
            'offset': note_event[1],
            'midi_value': note_event[2],
            'string': position[0], 
            'fret': position[1]
        })
    return predicted_notes
    
    
def compare(actual_notes, predicted_notes, time_threshold=0.1):
    """
    Compares the actual note with the predicted note by taking the time threshold into consideration.
    returns:
        results of true positive, false positive, false negative, string distance, fret distance and pitch matches
    """
    local_tp = 0
    local_fn = 0
    local_sd = 0
    local_fd = 0
    local_pitch_matches = 0
    
    matched_indices = set()
    
    for actual in actual_notes:
        best_match_idx = -1
        min_diff = float('inf')
        
        for i, pred in enumerate(predicted_notes):
            if i in matched_indices: continue
            
            diff = abs(actual['onset'] - pred['onset'])
            pitch_diff = abs(actual['midi_value'] - pred['midi_value'])
            if diff < time_threshold and pitch_diff < 1.0:
               if diff < min_diff:
                    min_diff = diff
                    best_match_idx = i
        
        if best_match_idx != -1:
            matched_indices.add(best_match_idx)
            pred = predicted_notes[best_match_idx]
            
            local_pitch_matches += 1
            
            ds, df = fret_string_match(actual, pred)
            local_sd += ds
            local_fd += df
            
            if ds == 0 and df == 0:
                # correct note and fret placement
                local_tp += 1
            else:
                # correct note, incorrect fret placement
                local_fn += 1 
        else:
            # incorrect note
            local_fn += 1
            
    local_fp = len(predicted_notes) - len(matched_indices)
    return local_tp, local_fp, local_fn, local_sd, local_fd, local_pitch_matches

def format_heuristic(note_events):
    """
    Matches the format of ground truth actual notes for ease
    """
    mapped_notes = all_midi_notes(note_events, STANDARD_TUNING)
    predicted_notes = []

    for item in mapped_notes:
        note_data = item[0]
        pos_data = item[1]
        
        predicted_notes.append({
            'onset': note_data[0],
            'offset': note_data[1],
            'midi_value': note_data[2],
            'string': pos_data[0], 
            'fret': pos_data[1]
        })
    return predicted_notes


def metric():
    if (all_true_positives + all_false_positives) > 0:
        precision = all_true_positives / (all_true_positives + all_false_positives)
    else: precision = 0
    if (all_true_positives + all_false_negatives) > 0:
        recall = all_true_positives / (all_true_positives + all_false_negatives)
    else: recall = 0
    if (precision + recall) > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else: f1 = 0
    return precision, recall, f1

def evaluate_process():
    """
    Generates the pipeline process for evaulate fretboard using guitarset
    """
    global all_true_positives, all_false_positives, all_false_negatives
    global total_string_distance, total_fret_distance, total_pitch_matches
    global total_actual_notes, total_predicted_notes

    # Define Paths
    path = r"backend/resources/annotation"
    path = os.path.join(BACKEND_ROOT, path)
    
    audio_path = r"backend/resources/audio_mono-mic"
    audio_path = os.path.join(BACKEND_ROOT, audio_path)

    jam_files = [f for f in listdir(path) if isfile(join(path, f))]
    test_set_files = [f for f in jam_files if f.startswith("05")]
    basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)
    
    # Exact path to your model file
    model_path = os.path.join(project_root, "backend", "models", "models", "best_fretboard_cnn.pt")

    try:
        mapper = FretBoardMapper(model_path=model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print(f"Scanning JAMS folders to find matches...")
    matches_found = 0

    
    for jam_path in test_set_files:
        print(f"Processing: {jam_path}")
        jam_file_path = join(path, jam_path)
        
        actual_notes = load_guitarset(jam_file_path)
        if not actual_notes:
            continue
        
        
        base_name = jam_path.split('.')[0] 
        audio_filename = base_name + '_mic.wav'
        audio_file_path = join(audio_path, audio_filename)
        
        
        
        try:
            # Using tuned parameters for GuitarSet to reduce ghost notes
            model_output, midi_data, note_events = predict(
                audio_file_path,
                basic_pitch_model,
                onset_threshold=0.6,
                frame_threshold=0.4,
                minimum_note_length=58.0,
                minimum_frequency=None,
                maximum_frequency=None
            )
        except Exception:
            continue
        
        note_events.sort(key=lambda x: x[0])
        
        # note_events = filter_process(note_events)
        
        predicted_notes = format_to_fretboard(note_events, mapper)
        # predicted_notes = format_heuristic(note_events)
        
        tp, fp, fn, sd, fd, pm = compare(actual_notes, predicted_notes)
        
        all_true_positives += tp
        all_false_positives += fp
        all_false_negatives += fn
        total_string_distance += sd
        total_fret_distance += fd
        total_pitch_matches += pm
        total_actual_notes += len(actual_notes)
        total_predicted_notes += len(predicted_notes)
        
        matches_found += 1
        if matches_found >= LIMIT: break

    if matches_found == 0:
        print("\nNo matches found. Ensure the JAMS and Audio folder names are identical.")
    
    precision, recall, f1 = metric()
    final_result(precision, recall, f1)

def final_result(precision, recall, f1):
    print("                        ~~~~Final Result For FretBoard Mapping~~~~                          ")
    print("=======================================================================")
    print(f"Total number of actual notes: {total_actual_notes}")
    print(f"Total number of predicted notes: {total_predicted_notes}")
    print("•••••")
    
    if total_pitch_matches > 0:
        fretboard_accuracy = all_true_positives / total_pitch_matches
        avg_string_error = total_string_distance / total_pitch_matches
        avg_fret_error = total_fret_distance / total_pitch_matches
    else:
        fretboard_accuracy = 0
        avg_string_error = 0
        avg_fret_error = 0
    print(f"Fretboard Exact Match Accuracy: {fretboard_accuracy*100:.4f}%")
    print(f"  (Of the notes properly detected, {fretboard_accuracy*100:.4f}% were placed on correct String/Fret)")
    print("•••••")
    print(f"Average String Distance Error:  {avg_string_error:.2f}")
    print(f"Average Fret Distance Error:    {avg_fret_error:.2f}")
    print("=======================================================================")
    print()

def main():
    evaluate_process()

if __name__ == '__main__':
    main()