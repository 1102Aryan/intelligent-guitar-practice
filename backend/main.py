from pathlib import Path
import argparse
import soundfile as sf
from core.audio_loader import audio_loader
from visualisation.plots import *
from core.pitch_detector import *
from tablature.fretboard_mapper import *
from tablature.tab_generator import *
from analysis.note_filters import *
from analysis.audio_seperation import *
from models.fretboard_mapper import FretBoardMapper

def automatic_music_transcription(file_path, tuning, min_confidence=0.5, min_duration=0.05, output_dir="outputs"):
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
        
        # Basic pitch detection ~ Spotify
        model_output, midi_data, note_events = pitch_detection(file_path)
        if not note_events:
            return "error: No notes detected in audio"
        
        # Processes the note events
        filtered_notes = filter_process(note_events)
        
        # Code for running FretBoardCNN
        # list of midi
        mapper = FretBoardMapper()
        midi_list = [note[2] for note in filtered_notes]
        mapped = mapper.map_notes(midi_list)

        # Create flat list of (note_event, position) tuples
        mapped_notes = []
        for note_event, position in zip(filtered_notes, mapped):
            mapped_notes.append((note_event, position)) 

        print(f"Created {len(mapped_notes)} mapped notes")
        
        # fretboard mapping
        # mapped_notes = all_midi_notes(filtered_notes, tuning)
        
        # if not mapped_notes:
        #     return "could not map notes to fretboard"
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
    result = automatic_music_transcription(file_path, current_tuning)
    print(result)
 
    
if __name__ == '__main__':
    main()    