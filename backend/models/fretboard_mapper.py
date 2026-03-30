import torch
import os
import sys
import numpy as np
from backend.models.synthtab.fretboard_cnn import FretBoardCNN



class FretBoardMapper:
    def __init__(self, model_path=r"backend/models/models/best_fretboard_cnn.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = FretBoardCNN(5, 32)
        #context = checkpoint.get('context_window', 5)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.open_strings = torch.tensor([64, 59, 55, 50, 45, 40], device=self.device)
        self.fallback_tuning = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40} 
    
    def _create_batch(self, note_tensor, window=5):
        # Pad with 0s so the first and last notes have context
        padded = torch.nn.functional.pad(note_tensor, (window, window), value=0)
        
        # Unfold creates  the sliding window view
        sequence_length = window * 2 + 1
        batch = padded.unfold(0, sequence_length, 1)
        
        return batch
    
    def _heuristic_fallback(self, midi_note):
        """
        Find the lowest playable fret for this note.
        """
        best_pos = (1, 0)
        min_fret = 99
        
        # Check all 6 strings
        for s_num, open_pitch in self.fallback_tuning.items():
            fret = midi_note - open_pitch
            if 0 <= fret <= 24:
                # Prefer lower frets
                if fret < min_fret:
                    min_fret = fret
                    best_pos = (s_num, int(fret))
        
        if min_fret == 99:
            # Note is too high/low
            return (1, 0) 
            
        return best_pos
           
        
    
    def _predict(self, contex_window):
        """
        Predicts correct string and fret from model
        """
        print(f"\n=== DEBUG _predict ===")
        print(f"Input shape: {contex_window.shape}")
        print(f"Input values: {contex_window}")
        
        with torch.no_grad():
            try:
                result = self.model(contex_window)
                print(f"Model output type: {type(result)}")
                print(f"Is tuple? {isinstance(result, tuple)}")
                
                if isinstance(result, tuple):
                    print(f"Tuple length: {len(result)}")
                    logits = result
                    string_logits, fret_logits = logits
                    string = torch.argmax(string_logits, dim=1).item() + 1
                    fret = torch.argmax(fret_logits, dim=1).item()
                    
                else:
                    print(f"ERROR: Got {type(result)} instead of tuple!")
                    print(f"Result: {result}")
                    return None, None
                
                
                
                print(f"Predicted string: {string}, fret: {fret}")
                return string, fret
                
            except Exception as e:
                print(f"ERROR in _predict: {e}")
                import traceback
                traceback.print_exc()
                return None, None
            
        
    def _get_context(self, notes, pos, window=5):
        """
        Finds the context window for each note adding padding (0) where required
        parameter:
            notes: contains the list of notes
            pos: the current position that is going to be predicted
            window: size of the values that can be added
        return:
            list of the items notes that fit the context window requirement
        """
        context_list = []
        for x in range(-window, window + 1):
            if (0 <= pos + x < len(notes)):
                context_list.append(notes[pos + x])
            else:
                context_list.append(0)
        return torch.tensor([context_list], dtype=torch.long)
    
    def _mask_by_midi(self, probs, midi_note):
        """
        Zero out classes whose string+fret doesn't produce the correct pitch.
        Same principle as GOAT's mask_by_pitch — forces the model to choose
        WHERE to play the note, not WHAT note to play.
        """
        masked = np.zeros_like(probs)
        for cls_idx in range(150):
            string_num = (cls_idx // 25) + 1
            fret_num = cls_idx % 25
            if self.fallback_tuning[string_num] + fret_num == midi_note:
                masked[cls_idx] = probs[cls_idx]
        # Fallback: note is outside guitar range — use raw probs
        if masked.sum() == 0:
            return probs
        return masked

    def _viterbi(self, probs):
        """
        Viterbi decoding over the sequence of per-note probability vectors.
        Physics penalties discourage large string changes and big fret jumps,
        producing a more playable tab than independent argmax.
        """
        T = len(probs)
        if T == 0:
            return []

        log_preds = np.log(np.array(probs) + 1e-9)
        path_prob = np.full((T, 150), -np.inf)
        backpointer = np.zeros((T, 150), dtype=int)

        path_prob[0] = log_preds[0]

        for t in range(1, T):
            prev = path_prob[t - 1]
            top_prev = np.argsort(prev)[-20:]

            for k in range(150):
                s_curr = (k // 25) + 1
                f_curr = k % 25
                best_score = -np.inf
                best_prev = 0

                for j in top_prev:
                    if prev[j] == -np.inf:
                        continue
                    s_prev = (j // 25) + 1
                    f_prev = j % 25

                    penalty = 0.0
                    if s_curr != s_prev:
                        penalty += 2.0
                    fret_dist = abs(f_curr - f_prev)
                    if fret_dist > 4:
                        penalty += 0.5 * (fret_dist - 4)

                    score = prev[j] + log_preds[t][k] - penalty
                    if score > best_score:
                        best_score = score
                        best_prev = j

                path_prob[t, k] = best_score
                backpointer[t, k] = best_prev

        best_path = []
        curr = int(np.argmax(path_prob[T - 1]))
        best_path.append(curr)
        for t in range(T - 1, 0, -1):
            curr = backpointer[t, curr]
            best_path.append(curr)
        best_path.reverse()
        return best_path

    def map_notes(self, midi_notes):
        """
        Maps a sequence of MIDI notes to (string, fret, confidence) tuples.

        Pipeline:
          1. Batch inference → softmax probabilities
          2. Pitch masking — constrain each note to valid string/fret positions
          3. Viterbi decoding — prefer playable sequences (same string, small fret jumps)
        """
        if not midi_notes:
            return []

        note_tensor = torch.tensor(midi_notes, dtype=torch.long)
        batch_input = self._create_batch(note_tensor).to(self.device)

        with torch.no_grad():
            logits = self.model(batch_input)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # (N, 150)

        # Constrain each note to positions that produce the correct pitch
        for i, midi_note in enumerate(midi_notes):
            probs[i] = self._mask_by_midi(probs[i], midi_note)

        best_path = self._viterbi(probs)

        results = []
        for i, idx in enumerate(best_path):
            string_num = (idx // 25) + 1
            fret_num = idx % 25
            confidence = float(probs[i][idx])
            results.append((string_num, fret_num, confidence))

        return results
                
