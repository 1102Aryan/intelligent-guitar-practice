from pathlib import Path
import argparse
import soundfile as sf
import os
import sys
import torch
import numpy as np
import librosa
from backend.core.audio_loader import audio_loader
from backend.visualisation.plots import *
from backend.core.pitch_detector import *
from backend.tablature.fretboard_mapper import *
from backend.models import fretboard_mapper
from backend.tablature.tab_generator import *
from backend.analysis.note_filters import *
from backend.analysis.audio_seperation import *
from backend.tablature.fretboard_mapper import all_midi_notes

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from backend.models.goat.goat_cnn import GoatFretboardCNN
except ImportError:
    sys.path.append(os.path.join(project_root, "backend", "models", "goat"))
MODEL_PATH = os.path.join(project_root, "backend", "models", "models", "goat_epoch_50.pth")
STANDARD_TUNING = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}


import numpy as np

def viterbi(predictions):
    """
    Viterbi decoding for guitar tab generation.
    input: predictions (T, 150) - raw probabilities
    output: list of best state indicies (0-149)
    """
    if not predictions or len(predictions) == 0:
        return []

    T = len(predictions)
    classes = 150
    
    # 1. Setup in logspace
    # Use -inf for initialization to represent 0 probability
    log_preds = np.log(np.array(predictions) + 1e-9)
    path_prob = np.full((T, classes), -np.inf)
    backpointer = np.zeros((T, classes), dtype=int)
    
    # Initialize first step
    path_prob[0] = log_preds[0]
    
    # Forward pass
    for t in range(1, T):
        prev_probs = path_prob[t-1]
        curr_emission = log_preds[t]
        
        # Optimisation: Look at top 20 previous candidates
        # If prev_probs is all -inf (start of silence) then handle gracefully
        top_prev_indices = np.argsort(prev_probs)[-20:] 
        
        for k in range(classes):
            s_curr, f_curr = (k // 25) + 1, (k % 25)
            
            best_score = float("-inf")
            best_prev = 0
            
            for j in top_prev_indices:
                # Skip impossible paths
                if prev_probs[j] == float("-inf"):
                    continue

                s_prev, f_prev = (j // 25) + 1, (j % 25)
                
                # Physics penalties
                penalty = 0.0
                
                # Penalty A - Changing strings (High cost)
                if s_curr != s_prev: 
                    penalty += 2.0 
                
                # Penalty B - Big fret jumps (Hand stretch)
                fret_dist = abs(f_curr - f_prev)
                if fret_dist > 4:
                    penalty += 0.5 * (fret_dist - 4)
                
                # Score calculation
                score = prev_probs[j] + curr_emission[k] - penalty
                
                if score > best_score:
                    best_score = score
                    best_prev = j
            
            path_prob[t, k] = best_score
            backpointer[t, k] = best_prev
            
    # Backward pass
    best_path = []
    
    # Start at the best ending state
    best_last_state = np.argmax(path_prob[T-1])
    best_path.append(best_last_state)
    
    curr = best_last_state
    for t in range(T-1, 0, -1):
        prev = backpointer[t, curr]
        best_path.append(prev)
        curr = prev
        
    best_path = best_path[::-1]
    
    return best_path




def get_goat_predictions(audio_path, model, device):
    """
    Runs the CNN on the audio file to get notes.
    """
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Detect Onsets Notes
    onset_f = librosa.onset.onset_detect(y=y, sr=sr, wait=1, pre_avg=3, post_avg=3, delta=0.05)
    onset_t = librosa.frames_to_time(onset_f, sr=sr)
    
    
    all_probs = []
    all_times = []
    mapped_notes = []
    MIN_WIDTH = 9

    for start_t in onset_t:
        # Slice Audio
        start_sample = int(start_t * sr)
        end_sample = start_sample + 4410
        if end_sample >= len(y): break
        
        audio_slice = y[start_sample:end_sample]
        
        # Normalise volume
        if np.max(np.abs(audio_slice)) > 0:
            audio_slice = audio_slice / np.max(np.abs(audio_slice))

        # Spectrogram
        spectrogram = librosa.feature.melspectrogram(y=audio_slice, sr=22050, n_mels=128)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        
        # Safety padding
        if spectrogram_db.shape[1] < MIN_WIDTH:
            pad_amt = MIN_WIDTH - spectrogram_db.shape[1]
            spectrogram_db = np.pad(spectrogram_db, ((0,0), (0, pad_amt)), mode='constant')

        tensor = torch.tensor(spectrogram_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(tensor)
            probs = torch.sigmoid(output).cpu().numpy().flatten()
            all_probs.append(probs)
            all_times.append(start_t)
            
        
    best_path_indices = viterbi(all_probs)
    for i, idx in enumerate(best_path_indices):
        confidence = all_probs[i][idx]
    
        # 5. Filter (Low threshold for high recall)
        if confidence > 0.15:
            string_num = (idx // 25) + 1
            fret_num = idx % 25
            
            midi_val = STANDARD_TUNING[string_num] + fret_num
            note_event = [all_times[i], all_times[i] + 0.2, midi_val, float(confidence)]
            
            mapped_notes.append((note_event, (string_num, fret_num)))

    return mapped_notes



def automatic_music_transcription(file_path, tuning, model_choice, min_confidence=0.5, min_duration=0.05, output_dir="outputs"):
    """
    Main function to handle transcription
    
    Perimeters:
        file_path: where the file is stored
        min_confidence: the minimum accepted confidence of audio
        min_duration: the minimum accepted duration of note
        output_dir: position of output file
    """
    try:
        audio_path = Path(file_path)
        if (not audio_path.exists()):
            return f"error: audio file not found: {audio_path}"
        
        audio_signal, sample_rate = audio_loader(audio_path)
        
        # Pre-processing
        # audio_signal, sample_rate = pre_process(audio_signal, sample_rate)
        
        # Converts back into wav file for Basic Pitch
        sf.write("processed.wav", audio_signal, sample_rate)
        file_path = "processed.wav"
        
        
        mapped_notes = []
        if model_choice == "End-to-End Architecture (GOAT Dataset)":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = GoatFretboardCNN().to(device)
            if os.path.exists(MODEL_PATH):
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                model.eval()
            else:
                return f"error: Model not found at {MODEL_PATH}"
            mapped_notes =  get_goat_predictions(file_path, model, device)
            
        elif model_choice == "Heuristic (Baseline)":
            # Basic pitch detection ~ Spotify
            model_output, midi_data, note_events = pitch_detection(file_path)
            if not note_events:
                return "error: No notes detected in audio"
            # Processes the note events
            filtered_notes = filter_process(note_events)
            
            # Code for running FretBoardCNN
            # list of midi
            print("Sample note:", filtered_notes[0])

            mapped_notes = all_midi_notes(filtered_notes, STANDARD_TUNING)

            # Create flat list of (note_event, position) tuples
            
            
        elif model_choice == "Sequential Architecture (SynthDataset)":
            # Basic pitch detection ~ Spotify
            model_output, midi_data, note_events = pitch_detection(file_path)
            if not note_events:
                return "error: No notes detected in audio"        
            # Processes the note events
            filtered_notes = filter_process(note_events)
            
            # Code for running FretBoardCNN
            # list of midi
            mapper = fretboard_mapper.FretBoardMapper()
            midi_list = [note[2] for note in filtered_notes]
            mapped = mapper.map_notes(midi_list)
 
            # Create flat list of (note_event, position) tuples
            mapped_notes = []
            for note_event, position in zip(filtered_notes, mapped):
                mapped_notes.append((note_event, position)) 
        else:
            return f"error: Invalid model choice selected: {model_choice}"
                

        print(f"Created {len(mapped_notes)} mapped notes")
        

        # sorted_notes = sorted_notes(mapped_notes)
        grouped_notes = group_notes(mapped_notes)
                
        # save_midi(audio_path)
        
        full_tab = Tab.display_ascii_tab(grouped_notes)
        return full_tab
    except Exception as e:
        return str(e)   

def main():
    """
    Command line interface
    """
    parser = argparse.ArgumentParser(
        description='CLI for Guitar AMT')
    parser.add_argument(
        "file_path",
        help="file path of audio file"    
    )
    audio_type = input("Is this a guitar only track (Y or N): ")

    tuning = int(input("Enter number for tuning (0: Standard, 1: Drop D, 2: Open C): "))
    current_tuning = TUNINGS[tuning]
    capo = int(input("Enter capo position (0-12): "))
    if (capo != 0):
        current_tuning = capo_position(capo, current_tuning)
        
    args = parser.parse_args()
    
    if audio_type.upper() == "N":
        # Sends to create guitar isolated track
        guitar_isolation(args.file_path)
        file_path = get_guitar_audio(args.file_path, "mdx_extra_q")
    else:
        file_path = args.file_path
    result = automatic_music_transcription(file_path, current_tuning, model_choice="End-to-End Architecture (GOAT Dataset)")
    print(result)
 
    
if __name__ == '__main__':
    main()    