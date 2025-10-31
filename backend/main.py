from pathlib import Path
import argparse

from core.audio_loader import audio_loader
from visualisation.plots import *
from core.pitch_detector import *
from tablature.fretboard_mapper import *
from tablature.tab_generator import *
from analysis.note_filters import *


def automatic_music_transcription(file_path, min_confidence=0.5, min_duration=0.05, output_dir="outputs"):
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
        # Basic pitch detection ~ Spotify
        model_output, midi_data, note_events = pitch_detection(file_path)
        if not note_events:
            return "error: No notes detected in audio"
        
        # Processes the note events
        filtered_notes = filter_process(note_events)
        
        # fretboard mapping
        mapped_notes = all_midi_notes(filtered_notes)
        
        if not mapped_notes:
            return "could not map notes to fretboard"
        # sorted_notes = sorted_notes(mapped_notes)
        grouped_notes = group_notes(mapped_notes)
        
                
        # save_midi(audio_path)
        
        Tab.display_ascii_tab(grouped_notes)
        return True
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
    
    args = parser.parse_args()
    result = automatic_music_transcription(args.file_path)
    print(result)
    

 
    
if __name__ == '__main__':
    main()    