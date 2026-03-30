import numpy as np
import librosa
import torch

from backend.models.goat.goat_cnn import GoatFretboardCNN
from backend.core.harmonic_cqt import compute_harmonic_cqt
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict, Model
basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)

STANDARD_TUNING = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

def viterbi(predictions):
    """
    Viterbi decoding for guitar tab generation.
    
    parameter:
        predictions (T, 150) - raw probabilities
    return:
        list of best state indicies (0-149)
    """
    if not predictions or len(predictions) == 0:
        return []

    T = len(predictions)
    classes = 150
    
    # Setup in logspace
    # Initalise as -inf as 0 probability
    log_preds = np.log(np.array(predictions) + 1e-9)
    path_prob = np.full((T, classes), -np.inf)
    backpointer = np.zeros((T, classes), dtype=int)
    
    # Initialise first step
    path_prob[0] = log_preds[0]
    
    # Forward pass
    for t in range(1, T):
        prev_probs = path_prob[t-1]
        curr_emission = log_preds[t]
        
        # Looks at the top 20 previous candidates
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
                
                # Penalty 1 due to changing strings High cost
                if s_curr != s_prev: 
                    penalty += 2.0 
                
                # Penalty 2 Big fret jumps Hand stretch
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


def mask_by_pitch(probs, detected_midi):
    """
    Zero out CNN predictions that don't produce the correct pitch.
    Forces the CNN to only decide which string/fret.
    return:
        masked list
    """
    masked = np.zeros_like(probs)
    
    for cls_idx in range(150):
        string_num = (cls_idx // 25) + 1
        fret_num = cls_idx % 25
        midi_val = STANDARD_TUNING[string_num] + fret_num
        
        if abs(midi_val - detected_midi) <= 0.5:
            masked[cls_idx] = probs[cls_idx]
    
    # Fallback use unmasked
    if masked.sum() == 0:
        return probs
    
    return masked

def filter_note_events(note_events, min_confidence=0.5, min_duration=0.05, guitar_range=(40, 84)):
    """
    Pre-filters Basic Pitch note events before CNN processing, matching the
    sequential pipeline's filter_process behaviour.
    - Removes low-confidence, Remove ghost notes, Removes redundancy, keeps highest confidence, Selects pitches to the playable guitar range
    """
    # Confidence & duration
    notes = [n for n in note_events
             if n[3] >= min_confidence and (n[1] - n[0]) >= min_duration]

    # Guitar range
    corrected = []
    lo, hi = guitar_range
    for n in notes:
        onset, offset, pitch, conf = n[0], n[1], n[2], n[3]
        while pitch < lo:
            pitch += 12
        while pitch > hi:
            pitch -= 12
        corrected.append((onset, offset, pitch, conf))

    # ascending Sort by onset then pitch
    corrected.sort(key=lambda x: (x[0], x[2]))

    # Remove duplicates -50 ms, keep highest confidence
    filtered = [corrected[0]] if corrected else []
    for note in corrected[1:]:
        prev = filtered[-1]
        if abs(note[0] - prev[0]) < 0.05 and note[2] == prev[2]:
            if note[3] > prev[3]:
                filtered[-1] = note
        else:
            filtered.append(note)

    return filtered


def get_goat_predictions(audio_path, model, device):

    y, sr = librosa.load(audio_path, sr=22050)

    # Basic Pitch selects notes
    model_output, midi_data, note_events = predict(
        audio_path,
        basic_pitch_model,
        onset_threshold=0.3,
        frame_threshold=0.2,
        minimum_note_length=30.0,
        minimum_frequency=None,
        maximum_frequency=None
    )

    if not note_events:
        return []

    # Post-processing
    note_events = filter_note_events(note_events)
    print(f"Notes after filtering: {len(note_events)}")

    if not note_events:
        return []

    all_probs = []
    all_times = []
    HALF_CONTEXT = int(0.5 * sr)

    for note in note_events:
        onset = note[0]
        offset = note[1]
        midi_pitch = note[2]

        center = int(onset * sr)
        win_start = center - HALF_CONTEXT
        win_end = center + HALF_CONTEXT

        if win_start < 0 or win_end >= len(y):
            continue

        audio_slice = y[win_start:win_end]
        if np.max(np.abs(audio_slice)) > 0:
            audio_slice = audio_slice / np.max(np.abs(audio_slice))

        harmonic_cqt = compute_harmonic_cqt(audio_slice, sr=sr)
        tensor = torch.tensor(harmonic_cqt, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.sigmoid(output).cpu().numpy().flatten()
            masked_probs = mask_by_pitch(probs, midi_pitch)
            all_probs.append(masked_probs)
            all_times.append((onset, offset, midi_pitch))

    best_path = viterbi(all_probs)

    mapped_notes = []
    for i, idx in enumerate(best_path):
        string_num = (idx // 25) + 1
        fret_num = idx % 25
        onset, offset, midi_pitch = all_times[i]
        midi_val = STANDARD_TUNING[string_num] + fret_num
        note_event = [onset, offset, midi_val, float(all_probs[i][idx])]
        mapped_notes.append((note_event, (string_num, fret_num)))

    return mapped_notes