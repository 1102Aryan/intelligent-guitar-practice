import jams
import soundfile as sf
import os
import sys
import numpy as np
from os import listdir
from os.path import isfile, join

# Path setup
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.models.fretboard_mapper import FretBoardMapper
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Global Counters
stats = {
    "total_actual": 0,
    "total_predicted": 0,
    "tp": 0,            
    "exact_matches": 0, 
    "string_dist": 0,
    "fret_dist": 0,
    # 7x7 Matrix (Index 0 unused, 1-6 used)
    "string_matrix": np.zeros((7, 7), dtype=int) 
}

LIMIT = 20

def load_guitarset(file_path):
    try:
        file = jams.load(file_path, strict=False)
    except:
        return []

    notes = []
    # GuitarSet Metadata Map
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
        
        if not hasattr(annotation.annotation_metadata, 'data_source'): continue
        ds = annotation.annotation_metadata.data_source
        if ds not in STRING_MAP: continue
        
        string_data = STRING_MAP[ds]
        open_pitch = string_data["open"]
        string_num = string_data["string_num"]

        for note in annotation:
            if note.value <= 0: continue
            
            fret = int(round(note.value - open_pitch))
            
            if isinstance(note.value, (int, float)): midi_val = note.value
            elif isinstance(note.value, dict): midi_val = note.value.get('value', 0)
            else: midi_val = 0
            
            if 0 <= fret <= 24:
                notes.append({
                    'onset': note.time,
                    'offset': note.time + note.duration,
                    'midi_value': midi_val,
                    'string': string_num, 
                    'fret': fret
                })
            
    notes.sort(key=lambda x: x['onset'])
    return notes

def format_to_fretboard(note_events, mapper):
    """
    Maps predictions to string/fret.
    FIXED: The CNN outputs the correct String Number (1-6) directly.
    We do NOT need to invert it.
    """
    if not note_events:
        return []
    
    midi_list = [note[2] for note in note_events]
    mapped = mapper.map_notes(midi_list)
    
    predicted_notes = []
    for note_event, position in zip(note_events, mapped):
        
        # --- THE FIX ---
        # Your CNN output (position[0]) is already the correct string number (1-6).
        # We just ensure it is an integer.
        guitar_string_num = int(position[0])
        
        predicted_notes.append({
            'onset': note_event[0],
            'offset': note_event[1],
            'midi_value': note_event[2],
            'string': guitar_string_num, 
            'fret': position[1]
        })
    return predicted_notes

def compare(actual_notes, predicted_notes, time_threshold=0.1):
    local_tp = 0
    local_exact = 0
    local_sd = 0
    local_fd = 0
    
    matched_indices = set()
    
    for actual in actual_notes:
        best_idx = -1
        min_diff = float('inf')
        
        for i, pred in enumerate(predicted_notes):
            if i in matched_indices: continue
            
            time_diff = abs(actual['onset'] - pred['onset'])
            pitch_diff = abs(actual['midi_value'] - pred['midi_value'])
            
            if time_diff < time_threshold and pitch_diff < 1.0:
                if time_diff < min_diff:
                    min_diff = time_diff
                    best_idx = i
        
        if best_idx != -1:
            matched_indices.add(best_idx)
            pred = predicted_notes[best_idx]
            
            local_tp += 1
            
            d_string = abs(actual['string'] - pred['string'])
            d_fret = abs(actual['fret'] - pred['fret'])
            
            local_sd += d_string
            local_fd += d_fret
            
            # --- POPULATE CONFUSION MATRIX ---
            # Rows = Actual, Cols = Predicted
            act_s = int(actual['string'])
            pred_s = int(pred['string'])
            
            # Safety check to stay in bounds
            if 1 <= act_s <= 6 and 1 <= pred_s <= 6:
                stats["string_matrix"][act_s][pred_s] += 1
            
            if d_string == 0 and d_fret == 0:
                local_exact += 1
            
    return local_tp, local_exact, local_sd, local_fd

def final_result():
    tp = stats["tp"]
    
    # Calc Metrics
    if tp > 0:
        exact_match_rate = stats["exact_matches"] / tp
        avg_string_error = stats["string_dist"] / tp
        avg_fret_error = stats["fret_dist"] / tp
    else:
        exact_match_rate = 0
        avg_string_error = 0
        avg_fret_error = 0

    print("\n")
    print("=======================================================================")
    print("                  FRETBOARD EVALUATION RESULTS                         ")
    print("=======================================================================")
    print(f"Notes Correctly Detected (TP): {tp}")
    print("-----------------------------------------------------------------------")
    print(f"Exact Fretboard Match Rate: {exact_match_rate*100:.2f}%")
    print(f"Avg String Error:           {avg_string_error:.2f}")
    print("-----------------------------------------------------------------------")
    
    print("\n[STRING CONFUSION MATRIX]")
    print("Rows = Actual String (1=High E, 6=Low E)")
    print("Cols = Predicted String")
    print("      P1   P2   P3   P4   P5   P6")
    
    for r in range(1, 7):
        row_str = f"A{r} |"
        for c in range(1, 7):
            val = stats["string_matrix"][r][c]
            row_str += f" {val:4}"
        print(row_str)
        
    print("\nInterpretation:")
    print("- Ideally, the highest numbers should be on the DIAGONAL (A1-P1, A2-P2...).")
    print("- If the diagonal goes Bottom-Left to Top-Right, your mapping is INVERTED.")
    print("=======================================================================")
    print()

def evaluate_process():
    path = os.path.join(BACKEND_ROOT, "backend/resources/annotation")
    audio_path = os.path.join(BACKEND_ROOT, "backend/resources/audio_mono-mic")
    
    # Model Path
    model_path = os.path.join(project_root, "backend", "models", "models", "best_fretboard_cnn.pt")

    try:
        basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)
        mapper = FretBoardMapper(model_path=model_path)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    jam_files = [f for f in listdir(path) if isfile(join(path, f))]
    matches_found = 0

    print(f"Processing up to {LIMIT} files...")

    for jam_path in jam_files:
        if matches_found >= LIMIT: break
        print(f"Processing: {jam_path}")
        
        jam_full = join(path, jam_path)
        actual_notes = load_guitarset(jam_full)
        if not actual_notes: continue
        
        base_name = jam_path.split('.')[0] 
        audio_full = join(audio_path, base_name + '_mic.wav')
        if not os.path.exists(audio_full): continue

        try:
            _, _, note_events = predict(
                audio_full,
                basic_pitch_model,
                onset_threshold=0.6,
                frame_threshold=0.4,
                minimum_note_length=58.0,
                minimum_frequency=None,
                maximum_frequency=None
            )
        except: continue
        
        note_events.sort(key=lambda x: x[0])
        predicted_notes = format_to_fretboard(note_events, mapper)
        
        tp, exact, sd, fd = compare(actual_notes, predicted_notes)
        
        stats["total_actual"] += len(actual_notes)
        stats["total_predicted"] += len(predicted_notes)
        stats["tp"] += tp
        stats["exact_matches"] += exact
        stats["string_dist"] += sd
        stats["fret_dist"] += fd
        
        matches_found += 1

    final_result()

def main():
    evaluate_process()

if __name__ == '__main__':
    main()