import jams
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from os.path import isfile, join
import torch
import os

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DatasetExtraction(Dataset):
   
    
    def __init__(self, file_path):
        """
        Extract jam files for training the model 
        """
        path = r"backend/resources/annotation"
        path = os.path.join(BACKEND_ROOT, path)
        
        audio_path = r"backend/resources/audio_mono-mic"
        audio_path = os.path.join(BACKEND_ROOT, audio_path)

        jam_files = [f for f in listdir(path) if isfile(join(path, f))]
        # audio_files = [a for a in listdir(audio_path) if isfile(join(audio_path, a))]
        
        for jam_path in jam_files[:5]:
            print(f"Processing: {jam_path}")
            jam_file_path = join(path, jam_path)
            actual_notes = load_guitarset(jam_file_path)
            base_name = jam_path.split('.')[0] 
            audio_filename = base_name + '_mic.wav'
            audio_file_path = join(audio_path, audio_filename)
        
    def pereprocess_dataset(dataset, batch_size=50, ):
        train_loader = DataLoader(dataset=dataset, shuffle=True)
        
    

