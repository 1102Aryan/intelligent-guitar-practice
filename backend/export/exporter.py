# https://pyguitarpro.readthedocs.io/en/stable/pyguitarpro/format.html#module-guitarpro.gp5
# https://stackoverflow.com/questions/53451286/python-reading-guitarpro-gp5-files
import guitarpro
from backend.export.tab_parser import TabParser

def export_to_gp5(tab_text, output_path="output.gp5"):
    """
    Exports tab text to Guitar Pro 5.
    """
    # Parse input
    parser = TabParser()
    parsed_notes = parser.parse(tab_text)
        
    # Create song
    song = guitarpro.Song()
    song.title = "AI Transcription"
    song.artist = "GOAT Model"
    song.tempo = 120
    
    if len(song.tracks) > 0:
        track = song.tracks[0]
    else:
        track = guitarpro.Track(song, guitarpro.MeasureHeader())
        song.tracks.append(track)
    
    #Set as Electric Guitar for tablature representation
    track.name = "Electric Guitar"
    track.color = guitarpro.Color(255, 0, 0)
    
    # MIDI settings
    track.channel.channel = 0 
    track.channel.instrument = 27
    
    # 6 string guitar tunining
    track.strings = [
        guitarpro.GuitarString(1, 64), # - High E
        guitarpro.GuitarString(2, 59), # - B
        guitarpro.GuitarString(3, 55), # - G
        guitarpro.GuitarString(4, 50), # - D
        guitarpro.GuitarString(5, 45), # - A
        guitarpro.GuitarString(6, 40)  # - Low E
    ]
    
    NOTES_PER_MEASURE = 8 
    num_measures = (len(parsed_notes) + NOTES_PER_MEASURE - 1) // NOTES_PER_MEASURE
    if num_measures < 1: num_measures = 1
    
    song.measureHeaders = [] 
    for _ in range(num_measures):
        header = guitarpro.MeasureHeader()
        header.timeSignature.numerator = 4
        header.timeSignature.denominator.value = 4
        song.measureHeaders.append(header)
        
    track.measures = [] 
    current_note_idx = 0
    
    for header in song.measureHeaders:
        measure = guitarpro.Measure(track, header)
        voice = measure.voices[0]
        
        for _ in range(NOTES_PER_MEASURE):
            if current_note_idx >= len(parsed_notes):
                break
            
            note_data = parsed_notes[current_note_idx]
            
            try:
                s_idx = int(note_data['string_idx'])
                fret_val = int(note_data['fret'])
                gp_string = s_idx + 1 
            except ValueError:
                current_note_idx += 1
                continue
            
            beat = guitarpro.Beat(voice)
            beat.duration.value = 8 
            note = guitarpro.Note(beat)
            note.value = fret_val
            note.velocity = 100
            note.string = gp_string
            note.type = guitarpro.NoteType.normal
            note.effect = guitarpro.NoteEffect()
            beat.notes.append(note)
            voice.beats.append(beat)
            current_note_idx += 1  
        track.measures.append(measure)

    # Save
    try:
        guitarpro.write(song, output_path)
        print(f"SUCCESS: Saved {output_path}")
        return output_path
    except Exception as e:
        print(f"FAILED to write GP5: {e}")
        return None