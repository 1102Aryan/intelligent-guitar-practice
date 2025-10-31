import librosa

STANDARD_TUNING = {
    6: 40, #E
    5: 45, #A
    4: 50, #D
    3: 55, #G
    2: 59, #B
    1: 64  #E
}

def note_to_frequency_mapping(note_list):
    # Using librosa to convert to Hz
    frequency_list = []
    for note in note_list:
        frequency_list.append(librosa.midi_to_hz(note))
    return frequency_list

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

def all_midi_notes(note_events):
    positions = []
    prev_pos = None
    for note in note_events:
        onset, offset, pitch_midi, velocity, confidence = note
        candidates = map_note_to_fret(pitch_midi)
        if not candidates:
            continue
        best = best_position(prev_pos, candidates)
        positions.append((note, best))
        prev_pos = best
    return positions
