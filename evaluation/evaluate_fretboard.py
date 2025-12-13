"""
Evaluates the fretboard using SynthTab
"""
import jams
import soundfile as sf
import os
import sys
import re

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.core.pitch_detector import pitch_detection
from backend.analysis.note_filters import filter_process
from backend.models.fretboard_mapper import FretBoardMapper

# Global Counters
all_true_positives = 0
all_false_negatives = 0
all_false_positives = 0
total_actual_notes = 0
total_predicted_notes = 0
total_string_distance = 0
total_fret_distance = 0
LIMIT = 20  # Stops after finding this many MATCHES

def normalize_folder_name(name):
    """
    Simple normalization to handle Windows/Linux case differences.
    """
    return name.strip().lower()

def index_audio_files(audio_root_path):
    """
    Creates a map of Audio Files based on their PARENT FOLDER.
    Key: Normalized Folder Name
    Value: Full Path to Audio File
    """
    print(f"Indexing audio files in: {audio_root_path}...")
    audio_map = {}
    for root, dirs, files in os.walk(audio_root_path):
        for f in files:
            if f.endswith(('.wav', '.flac', '.mp3')):
                # Key is the folder name (e.g. "_MaTrioK_ - Kazowwie...__midi")
                folder_name = os.path.basename(root)
                norm_key = normalize_folder_name(folder_name)
                
                # Store full path
                audio_map[norm_key] = os.path.join(root, f)
    
    print(f"-> Found {len(audio_map)} audio tracks.")
    return audio_map

def extract_synthTab(file_path):
    notes = []
    try:
        # Disable validation for 'note_tab'
        jam = jams.load(file_path, validate=False)
        tuning_map = {64: 1, 59: 2, 55: 3, 50: 4, 45: 5, 40: 6}
        bpm = load_tempo(file_path)
        # MIDI resolution for guitar pro
        ticks = 960
        second_per_tick = 60 / (bpm * ticks)
        
        for ann in jam.annotations:
            if ann.namespace != 'note_tab':
                continue
            
            # Use Sandbox for tuning
            sandbox = ann.sandbox
            open_tuning = getattr(sandbox, 'open_tuning', None)
            
            if open_tuning in tuning_map:
                string_num = tuning_map[open_tuning]
            else:
                continue

            for obs in ann.data:
                # Handle value types safely
                if isinstance(obs.value, dict):
                    fret = obs.value.get('fret')
                else:
                    fret = obs.value
                
                onset_time = obs.time * second_per_tick
                
                if fret is not None:
                    notes.append({
                        'onset': onset_time,
                        'offset': obs.time + obs.duration,
                        'string': int(string_num),
                        'fret': int(fret)
                    })
        notes.sort(key=lambda x: x['onset'])
        return notes
    except Exception:
        return []

def fret_string_match(actual_note, predicted_note):
    diff_string = abs(actual_note['string'] - predicted_note['string'])
    diff_fret = abs(actual_note['fret'] - predicted_note['fret'])
    return diff_string, diff_fret

def load_tempo(file):
    folder = os.path.dirname(file)
    file_path = os.path.join(folder, "tempo.txt")
    if os.path.exists(file_path):
        try: 
            with open(file_path, 'r') as f:
                content = f.read().strip()
                return float(content)
        except:
            pass
    return 120

def get_cnn_predicted_notes(audio_path, mapper):
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

# def compare(actual_notes, predicted_notes, time_threshold=0.1):
#     local_tp = 0
#     local_fn = 0
#     local_sd = 0
#     local_fd = 0
#     matched_indices = set()
    
#     for actual in actual_notes:
#         best_idx = -1
#         min_diff = float('inf')
        
#         for i, pred in enumerate(predicted_notes):
#             if i in matched_indices: continue
            
#             diff = abs(actual['onset'] - pred['onset'])
#             if diff < time_threshold and diff < min_diff:
#                 min_diff = diff
#                 best_idx = i
        
#         if best_idx != -1:
#             matched_indices.add(best_idx)
#             pred = predicted_notes[best_idx]
#             ds, df = fret_string_match(actual, pred)
#             local_sd += ds
#             local_fd += df
            
#             if ds == 0 and df == 0:
#                 local_tp += 1
#             else:
#                 local_fn += 1 
#         else:
#             local_fn += 1
            
#     local_fp = len(predicted_notes) - len(matched_indices)
#     return local_tp, local_fp, local_fn, local_sd, local_fd

def compare(actual_notes, predicted_notes, time_threshold=0.1):
    local_tp = 0
    local_fn = 0
    local_sd = 0
    local_fd = 0
    
    matched_indices = set()
    
    # DEBUG: Check Time Units
    if actual_notes and predicted_notes:
        print(f"   [DEBUG] JAMS Time: {actual_notes[0]['onset']} | PRED Time: {predicted_notes[0]['onset']}")
    
    for actual in actual_notes:
        best_match_idx = -1
        min_diff = float('inf')
        
        for i, pred in enumerate(predicted_notes):
            if i in matched_indices: continue
            
            diff = abs(actual['onset'] - pred['onset'])
            if diff < time_threshold and diff < min_diff:
                min_diff = diff
                best_match_idx = i
        
        if best_match_idx != -1:
            matched_indices.add(best_match_idx)
            pred = predicted_notes[best_match_idx]
            
            ds, df = fret_string_match(actual, pred)
            local_sd += ds
            local_fd += df
            
            if ds == 0 and df == 0:
                local_tp += 1
            else:
                local_fn += 1 
        else:
            local_fn += 1
            
    local_fp = len(predicted_notes) - len(matched_indices)
    return local_tp, local_fp, local_fn, local_sd, local_fd


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
    global all_true_positives, all_false_positives, all_false_negatives
    global total_string_distance, total_fret_distance
    global total_actual_notes, total_predicted_notes

    # Define Paths
    jams_root = os.path.join(project_root, "backend", "resources", "all_jams_midi_V2_60000_tracks", "outall")
    audio_root = os.path.join(project_root, "backend", "resources", "lespaul_clean_bridge", "lespaul_clean_bridge")
    # Exact path to your model file
    model_path = os.path.join(project_root, "backend", "models", "models", "best_fretboard_cnn.pt")

    try:
        mapper = FretBoardMapper(model_path=model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 1. Index Audio Files (Smaller dataset)
    audio_map = index_audio_files(audio_root)
    
    print(f"Scanning JAMS folders to find matches...")
    matches_found = 0

    # 2. Iterate JAMS folders
    for root, dirs, files in os.walk(jams_root):
        if matches_found >= LIMIT: break
        
        for f in files:
            if f.endswith(".jams"):
                # Check if this JAMS folder exists in our Audio Map
                # Folder Name: "_MaTrioK_ - Kazowwie - gp3__1 - Distortion Guitar__midi"
                folder_name = os.path.basename(root)
                norm_key = normalize_folder_name(folder_name)
                
                if norm_key in audio_map:
                    # MATCH FOUND!
                    print(f"[{matches_found+1}/{LIMIT}] Match: {folder_name}")
                    
                    jams_path = os.path.join(root, f)
                    audio_path = audio_map[norm_key]
                    
                    # 3. Process the Pair
                    actual_notes = extract_synthTab(jams_path)
                    if not actual_notes:
                        print("   -> Skipping (No notes in JAMS)")
                        continue
                        
                    predicted_notes = get_cnn_predicted_notes(audio_path, mapper)
                    
                    tp, fp, fn, sd, fd = compare(actual_notes, predicted_notes)
                    
                    all_true_positives += tp
                    all_false_positives += fp
                    all_false_negatives += fn
                    total_string_distance += sd
                    total_fret_distance += fd
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
    print(f"True Positives: {all_true_positives}")
    print(f"False Positives: {all_false_positives}")
    print(f"False Negatives: {all_false_negatives}")
    print("•••••")
    print(f"The precision of the model: {precision:.4f}")
    print(f"The recall of the model: {recall:.4f}")
    print(f"The F1-Score of the model: {f1:.4f}")
    print(f"How far off from actual note, fret {total_string_distance} and string: {total_fret_distance}")
    print("=======================================================================")
    print()

def main():
    evaluate_process()

if __name__ == '__main__':
    main()