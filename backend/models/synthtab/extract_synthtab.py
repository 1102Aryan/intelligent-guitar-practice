import torch
from torch.utils.data import DataLoader, Dataset
import os
import jams
import numpy as np

class ExtractSynthTab(Dataset):
    """
    Dataset class to extract guitar tabs from JAMS files.
    """
    def __init__(self, file_path, context_window=5, max_files=10):
        self.context_window = context_window
        self.samples = []
        
        self.tuning_to_string = {
            64: 0,  # High E
            59: 1,  # B
            55: 2,  # G
            50: 3,  # D
            45: 4,  # A
            40: 5   # Low E
        }

        # Path setup
        abs_path = os.path.abspath(file_path)
        print(f"Scanning directory: {abs_path}")
        
        # Searching JAM files
        jam_files = []
        for root, dirs, files in os.walk(abs_path):
            for f in files:
                if f.endswith(".jams"):
                    jam_files.append(os.path.join(root, f))
        # shortens to max files number
        if max_files:
            jam_files = jam_files[:max_files]
            
        print(f"Found {len(jam_files)} JAMS files. Processing...")

        # Collect Information
        for i, jam_path in enumerate(jam_files):
            if i % 10 == 0 and i > 0:
                print(f"  Processed {i}/{len(jam_files)} files...")
            self._extract_from_jams(jam_path)
            
        print(f"Extraction complete! Total samples found: {len(self.samples)}")

    def _extract_from_jams(self, jam_path):
        """
        Extracts notes from JAMS file using note_tab namespace.
        Handles validation=False and getattr for Sandbox.
        """
        try:
            # Load files
            jam = jams.load(jam_path, validate=False)
            all_notes = []

            for annotation in jam.annotations:
                # Filter for tab data
                if annotation.namespace != 'note_tab':
                    continue
                
                sandbox = annotation.sandbox
                open_tuning = getattr(sandbox, 'open_tuning', None)
                
                if open_tuning is None:
                    s_idx = getattr(sandbox, 'string_index', None)
                    if s_idx is not None:
                    
                        fallback_map = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
                        # If keys are 0-5 instead:
                        if int(s_idx) == 0: fallback_map = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
                        
                        open_tuning = fallback_map.get(int(s_idx))

                if open_tuning not in self.tuning_to_string:
                    continue
                    
                string_idx = self.tuning_to_string[open_tuning]

                # Extract notes
                for obs in annotation.data:
                    fret = None

                    if isinstance(obs.value, dict):
                        fret = obs.value.get('fret')
                    elif hasattr(obs.value, 'fret'):
                        fret = getattr(obs.value, 'fret')
                    elif isinstance(obs.value, (int, float)):
                        fret = obs.value
                    
                    if fret is None:
                        continue
                        
                    # Calculate MIDI pitch based on Open String + Fret
                    midi_pitch = int(open_tuning + fret)
                    
                    if 0 <= fret <= 24:
                        all_notes.append({
                            'midi': midi_pitch,
                            'string': string_idx,
                            'fret': int(fret),
                            'time': obs.time 
                        })

            # ascennding order based on onset time
            all_notes.sort(key=lambda x: x['time'])

            # Create context windows 
            for i in range(len(all_notes)):
                context = self._get_context(all_notes, i)
                self.samples.append({
                    'context': context,
                    'string': all_notes[i]['string'],
                    'fret': all_notes[i]['fret']
                })
        # Catch error if fails
        except Exception as e:
            print(f"Error reading {os.path.basename(jam_path)}: {e}")

    def _get_context(self, notes, idx):
        """
        Get context window around current note
        return:
            context
        """
        context = []
        # Previous notes
        for i in range(idx - self.context_window, idx):
            if i >= 0:
                context.append(notes[i]['midi'])
            else:
                context.append(0)
        
        # Current note
        context.append(notes[idx]['midi'])
        
        # Next notes
        for i in range(idx + 1, idx + self.context_window + 1):
            if i < len(notes):
                context.append(notes[i]['midi'])
            else:
                context.append(0)
        
        return context
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample['context'], dtype=torch.long),
            torch.tensor(sample['string'], dtype=torch.long),
            torch.tensor(sample['fret'], dtype=torch.long)
        )

def create_data_loader(file_dir, batch_size, train_split=0.8, context_window=5, max_files=10):
    full_dataset = ExtractSynthTab(file_dir, context_window, max_files)
    
    if len(full_dataset) == 0:
        raise ValueError("No samples found. Check path or JAMS structure.")
    
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader