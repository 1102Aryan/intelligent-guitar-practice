import pretty_midi
import scipy.io.wavfile
import numpy as np
from backend.export.tab_parser import TabParser

def export_to_midi(tab_text, output_path="temp_midi.mid"):
    """
    Parses tab and genertes a playable midi
    """
    parser = TabParser()
    parsed_notes = parser.parse(tab_text)
    
    if not parsed_notes:
        print("No notes to play.")
        return None
    
    midi = pretty_midi.PrettyMIDI()
    
    guitar_program = pretty_midi.instrument_name_to_program('Electric Guitar (Clean)')
    guitar = pretty_midi.Instrument(program=guitar_program)
    string_bases = [64, 59, 55, 50, 45, 40]
    
    current_time = 0.0
    note_duration = 0.25
    
    for d in parsed_notes:
        try:
            string_idx = int(d['string_idx'])
            fret = int(d['fret'])
        except ValueError:
            continue 

        if 0 <= string_idx < 6:
            # Calculate the actual musical pitch
            pitch = string_bases[string_idx] + fret
            
            # Note object
            note = pretty_midi.Note(
                velocity=100,
                pitch=pitch, 
                start=current_time, 
                end=current_time + note_duration
            )
            guitar.notes.append(note)
            
        # Move the timeline forward
        current_time += note_duration
        
    # Save file
    midi.instruments.append(guitar)
    midi.write(output_path)
    return output_path
    
def export_to_audio(tab_text, output_path="temp_midi.wav"):
    midi_path = export_to_midi(tab_text)
    if not midi_path:
        return None
    # Normailse audio
    midi_obj = pretty_midi.PrettyMIDI(midi_path)
    audio_data = midi_obj.synthesize(fs=44100)
    audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    # Save as wav file
    scipy.io.wavfile.write(output_path, 44100, audio_data)
    return output_path
    
    