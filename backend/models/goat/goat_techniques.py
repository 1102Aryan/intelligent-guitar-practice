import os
import guitarpro
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

def extract_goat_techniques(file_path, output_dir, clip_duration=0.2):
    """
    Parses GOAT .gp5 files to extract audio clips of specific techniques
    """
    techniques = ['normal', 'bend', 'slide', 'vibrato']
    for t in techniques:
        os.makedirs(os.path.join(output_dir, t), exist_ok=True)
        
    counts = {k: 0 for k in techniques}
    MAX_PER_CLASS = 2000
    
    print(f"Scanning GOAT from {file_path}...")
    
    