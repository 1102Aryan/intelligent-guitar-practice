import librosa
# Maps note to fretboard
# f = 440 . 2^(n-69)/12
# https://inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
# Midi

STANDARD_TUNING = {
    6: 40, #E2
    5: 45, #A2
    4: 50, #D3
    3: 55, #G3
    2: 59, #B3
    1: 64  #E4
}

DROPD_TUNING = {
    6: 38, #D2
    5: 45, #A2
    4: 50, #D3
    3: 55, #G3
    2: 59, #B3
    1: 64, #E4
}

OPENC_TUNING = {
    6: 36, #C2
    5: 43, #G2
    4: 48, #C3
    3: 55, #G3
    2: 60, #C4
    1: 64, #E4 
}

TUNINGS = {
    0: STANDARD_TUNING,
    1: DROPD_TUNING,
    2: OPENC_TUNING,
}

def note_to_frequency_mapping(note_list):
    """
    Using librosa to convert to Hz
    Parameters:
        note_list
    Return:
        returns a list of the frequency from the midi
    """
    frequency_list = []
    for note in note_list:
        frequency_list.append(librosa.midi_to_hz(note))
    return frequency_list

def capo_position(capo_no, tuning = STANDARD_TUNING):
    """
    Updates the tuning based on the position of the capo. 1 capo = 1 semitone up
    Parameter:
        capo_no: the position the capo is set to
        tuning: the current pitches of the open strings in the guitar
    Return:
        returns a new tuning based on the capo position
    """
    new_tuning = {}
    for string_no, midi_pitch in tuning.items():
        new_pitch = midi_pitch + capo_no
        new_tuning[string_no] = new_pitch
    return new_tuning
            

def map_note_to_fret(note_midi, max_fret = 24, tuning = STANDARD_TUNING):
    # fret = midi_note - open_string
    positions = []
    for string, open_midi in tuning.items():
        fret = int(note_midi) - open_midi
        if 0 <= fret <= max_fret:
            positions.append((string, fret))
    return positions

def best_position(prev_pos, curr_pos):
    if not prev_pos:
        return min(curr_pos, key=lambda x: abs(x[1]-5)) # Middle of fretboard
    prev_string, prev_fret = prev_pos
    return min(curr_pos, key=lambda x: abs(x[0]-prev_string) + abs(x[1]-prev_fret))

def all_midi_notes(note_events, tuning_type):
    """
    Maps all the notes to the fretboard.
    Perimeter:
        note_event
        tuning_no: Used to select which tuning user picked
    returns 
        note events with fretboard mapping  
    """
    # tuning_type = TUNINGS.get(tuning_no, STANDARD_TUNING)
    positions = []
    prev_pos = None
    for note in note_events:
        onset, offset, pitch_midi, velocity, confidence = note
        candidates = map_note_to_fret(pitch_midi, tuning=tuning_type)
        if not candidates:
            continue
        best = best_position(prev_pos, candidates)
        positions.append((note, best))
        prev_pos = best
    return positions
