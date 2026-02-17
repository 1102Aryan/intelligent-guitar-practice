import torch
import os
import sys
from .fretboard_cnn import FretBoardCNN



class FretBoardMapper:
    def __init__(self, model_path=r"backend/models/models/best_fretboard_cnn.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = FretBoardCNN(5, 32)
        #context = checkpoint.get('context_window', 5)
        self.model.load_state_dict(checkpoint['model_state_dict'])
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
    
    def map_notes(self, midi_notes):
        """
        Runs the process for mapping notes with heuristic safety net
        """
        if not midi_notes:
            return []
        
        note_tensor = torch.tensor(midi_notes, dtype=torch.long)
        batch_input = self._create_batch(note_tensor)
        batch_input = batch_input.to(self.device)
        
    
        results = []
        with torch.no_grad():
            logits = self.model(batch_input)
            pred_indices = torch.argmax(logits, dim=1)
        
        pred_indices = pred_indices.cpu().numpy()
        input_pitches = note_tensor.cpu().numpy()
        open_strings_np = self.open_strings.cpu().numpy()

        for i, note in enumerate(midi_notes):
            classes = pred_indices[i]
            ml_string_idx = classes // 25 
            ml_fret = classes % 25
            ml_pos = (int(ml_string_idx + 1), int(ml_fret))
            ml_predicted_pitch = open_strings_np[ml_string_idx] + ml_fret
            is_ml_valid = (ml_predicted_pitch == note)
            heuristic_pos = self._heuristic_fallback(note)
            (h_string, h_fret) = heuristic_pos

            final_pos = ml_pos

            if not is_ml_valid:
                final_pos = heuristic_pos
            elif h_fret == 0:
                if ml_fret > 4: 
                    final_pos = heuristic_pos
            elif h_fret <= 3 and ml_fret > 7:
                final_pos = heuristic_pos

            results.append(final_pos)

        return results
                
