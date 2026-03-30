import jams
import soundfile as sf
import librosa
import os
from os import listdir
from os.path import isfile, join
import sys
import re
from tqdm import tqdm
import numpy as np
import torch

# Setting up path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# File path
AUDIO_ROOT = os.path.join(project_root, "backend", "resources", "audio_mono-mic")
# Jam path
JAM_ROOT = os.path.join(project_root, "backend", "resources", "annotation")
# Model path
MODEL_PATH = os.path.join(project_root, "backend", "models", "models", "goat_3.pth")

 


from backend.models.goat.goat_cnn import GoatFretboardCNN
from backend.models.goat.goat_prediction import get_goat_predictions



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

def get_cnn_predicted_notes(audio_path, model, device):
    """
    Use GOAT fretboard CNN to predict the string and fret (End-to-End system).
    
    returns:
        predicted list of onset, offset, string, fret
    """
    if not os.path.exists(audio_path):
        return []

    predicted_notes_goat = get_goat_predictions(audio_path, model, device)
    predicted_notes = []
    for note_events, (string_num, fret_num) in predicted_notes_goat:            
            predicted_notes.append({
                'onset': note_events[0],
                'midi_value': note_events[2],
                'string': string_num,
                'fret': fret_num
            })
    return predicted_notes    
    
def compare(actual_notes, predicted_notes, time_threshold=0.1):
    """
    Compares the actual note with the predicted note by taking the time threshold into consideration.
    returns:
        results of true positive, false positive, false negative, string distance, fret distance and pitch matches
    """
    local_tp = 0
    local_fp = 0
    local_fn = 0
    local_sd = 0
    local_pitch_tp = 0
    local_fd = 0
    local_pitch_matches = 0
    
    matched_prediction_indices = set()
    matched_actual_indices = set()
    
    for x, actual in enumerate(actual_notes):
        best_match_idx = -1
        min_diff = float('inf')
        
        for i, pred in enumerate(predicted_notes):
            if i in matched_prediction_indices:
                continue
            
            diff = abs(actual['onset'] - pred['onset'])
            pitch_diff = abs(actual['midi_value'] - pred['midi_value'])
            if diff < time_threshold and pitch_diff < 1.0:
               if diff < min_diff:
                    min_diff = diff
                    best_match_idx = i
        
        if best_match_idx != -1:
            matched_prediction_indices.add(best_match_idx)
            matched_actual_indices.add(x)
            pred = predicted_notes[best_match_idx]
            
            local_pitch_matches += 1
            
            ds, df = fret_string_match(actual, pred)
            local_sd += ds
            local_fd += df
            
            if ds == 0 and df == 0:
                # correct note and fret placement
                local_tp += 1

    local_fn = len(actual_notes) - len(matched_actual_indices)
    local_fp = len(predicted_notes) - len(matched_prediction_indices)
    return local_tp, local_fp, local_fn, local_sd, local_fd, local_pitch_matches

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading GOAT Model on {device}...")
    model = GoatFretboardCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print(f"Scanning JAMS folders to find matches...")
    
    jam_files = [f for f in listdir(JAM_ROOT) if f.endswith('.jams') and f.startswith("05")]

    print(f"Found {len(jam_files)}!")

    
    matches_found = 0

    
    for jam_path in tqdm(jam_files):
        print(f"Processing: {jam_path}")
        jam_file_path = join(JAM_ROOT, jam_path)
        
        actual_notes = load_guitarset(jam_file_path)
        if not actual_notes:
            continue
        
        
        base_name = jam_path.replace(".jams", "")
        audio_filename = base_name + '_mic.wav'
        audio_file_path = join(AUDIO_ROOT, audio_filename)
        
        if not os.path.exists(audio_file_path):
            print(f"Missing audio: {audio_file_path}")
            continue
        
        
        
        predicted_notes = get_cnn_predicted_notes(audio_file_path, model, device)
        
        
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
    print("                        ~~~~FINAL RESULTS FOR GOAT CNN~~~~                          ")
    print("=======================================================================")
    print(f"Total number of actual notes: {total_actual_notes}")
    print(f"Total number of predicted notes: {total_predicted_notes}")
    print(f"Pitch matches (correct note, any position): {total_pitch_matches}")
    print("•••••")
    
    pitch_recall = total_pitch_matches / total_actual_notes if total_actual_notes > 0 else 0
    print(f"Pitch Detection Recall: {pitch_recall*100:.2f}%")
    print(f"  ({total_pitch_matches} of {total_actual_notes} actual notes detected)")
    print()
    if total_pitch_matches > 0:
        fretboard_accuracy = all_true_positives / total_pitch_matches
        avg_string_error = total_string_distance / total_pitch_matches
        avg_fret_error = total_fret_distance / total_pitch_matches
    else:
        fretboard_accuracy = 0
    print(f"Fretboard Exact Match Accuracy: {fretboard_accuracy*100:.4f}%")
    print(f"  (Of the notes properly detected, {fretboard_accuracy*100:.4f}% were placed on correct String/Fret)")
    print("•••••")
    print(f"Average String Distance Error:  {avg_string_error:.2f}")
    print(f"Average Fret Distance Error:    {avg_fret_error:.2f}")
    print("=======================================================================")
    print()
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")


def main():
    evaluate_process()

if __name__ == '__main__':
    main()