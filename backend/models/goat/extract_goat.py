import os
import numpy as np
import glob
import librosa
import pandas as pd
import guitarpro as gp
import soundfile as sf
from tqdm import tqdm
from core.harmonic_cqt import compute_harmonic_cqt


class GOATProcessor:
    def __init__(self, dataset_path, output_dir):
        self.output_dir = output_dir
        self.metadata_csv = os.path.join(output_dir, 'metadata.csv')
        self.dataset_path = dataset_path
        self.sr = 22050
        self.hop_length = 512
        
        os.makedirs(os.path.join(self.output_dir, "specs"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels"), exist_ok=True)
        
    def load_metadata(self):
        return pd.read_csv(self.metadata_csv)
    
    def process_dataset(self, limit=5):
        """
        Finds all the pairs of .wav and .gp5, processing them
        """
        
        # Find all folders starting with "item_" name
        item_folders = [
            f for f in os.listdir(self.dataset_path) 
            if os.path.isdir(os.path.join(self.dataset_path, f)) and f.startswith("item_")
        ]
        
        # Sort them numerically so item_2 comes after item_1
        item_folders.sort(key=lambda x: int(x.split('_')[1]) if '_' in x else 0)

        if not item_folders:
            print(f"ERROR: No 'item_X' folders found in {self.dataset_path}")
            return
        
        print(f"Processing data from {self.dataset_path}")
      
        processed_count = 0
        
        if limit:
            files = item_folders[:limit]
            print(f"Test Run: Processing only {limit} files")
        
        for item_name in tqdm(files, desc="Processing Songs"):
            item_dir = os.path.join(self.dataset_path, item_name)
            gp_files = glob.glob(os.path.join(item_dir, "*.gp5"))
            if not gp_files:
                continue 
            gp_path = gp_files[0]
            
            wav_files = glob.glob(os.path.join(item_dir, "*.wav"))
            audio_path = None
            
            if not wav_files:
                continue

            # Picks clean wavfile, else takes the mic
            for w in wav_files:
                if "clean" in w: audio_path = w; break
            if not audio_path:
                for w in wav_files:
                    if "mic" in w: audio_path = w; break
            if not audio_path:
                audio_path = wav_files[0]

            # Process
            try:
                self.process_item(item_name, audio_path, gp_path)
                processed_count += 1
            except Exception as e:
                print(f"Skipping {item_name} due to error: {e}")

        print(f"Done! Processed {processed_count} songs.")
    
    def process_item(self, item_name, audio_path, gp_path):
        y, _ = librosa.load(audio_path, sr=self.sr)
        
        try:
            song = gp.parse(gp_path)
        except Exception as e:
            print(f"Error parsing {gp_path}: {e}")
            return
        # 22050 SR
        CONTEXT_SAMPLES = int(1.0 * self.sr) 
        # 11025 half of SR 
        HALF_CONTEXT = CONTEXT_SAMPLES // 2  
        
        current_time = 0.0
        tempo = song.tempo
        ticks_per_quarter = 960
        
        events = []
        
        for track in song.tracks:
            for measure in track.measures:
                seconds_per_tick = 60.0 / tempo / ticks_per_quarter
                
                voice = measure.voices[0]
                for beat in voice.beats:
                    duration = beat.duration.time * seconds_per_tick
                    
                    for note in beat.notes:
                        events.append({
                            "time": current_time,
                            "string": note.string,
                            "fret": note.value
                        })
                    current_time += duration
            break
        
        for i, event in enumerate(events):
            center = int(event["time"] * self.sr)
            start_sample = center - HALF_CONTEXT
            end_sample = center + HALF_CONTEXT 
            
            if 0 > start_sample or end_sample < len(y):
                audio_slice = y[start_sample:end_sample]
                
                harmonic_cqt = compute_harmonic_cqt(audio_slice, sr=self.sr)
                
                # Save Data
                # ID format: songname_index
                file_id = f"{item_name}_{i}"
                
                np.save(os.path.join(self.output_dir, "specs", f"{file_id}.npy"), harmonic_cqt)
                
                # Label: [String (1-6), Fret (0-24)]
                label = np.array([event["string"], event["fret"]])
                np.save(os.path.join(self.output_dir, "labels", f"{file_id}.npy"), label)
        
        