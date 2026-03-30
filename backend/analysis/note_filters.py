def confidence_filter(note_events, min=0.5):
    """
    Filters out all the notes that have a low confidence level.
    
    Parameters:
        note_events: list from the basic pitch (onset, offset, pitch, velocity, confidence)
        min: minimum confidence threshold that will get accepted
    
    Returns:
        returns notes that meet the requirement
    """
    notes = []
    for note in note_events:
        if (note[3] >= min):
            notes.append(note)
    return notes

def short_note_filter(note_events, min=0.05):
    """
    Removes all the short notes from the note events based on difference of offset (note[1]) and onset (note[0]).
    
    Parameters:
        note_events
        min: the minimum duration a note can last.
        
    """
    notes = []
    for note in note_events:
        if (note[1] - note[0] >= min):
            notes.append(note)
    return notes

def duplicate_note_filter(note_events, min=0.05):
    """
    Removes all duplicates notes from the note events based on min (minimum time threshold).
    
    Parameters:
        note_events
        min: the minimum duration that a note must not repeat in 

    """
    merge = [note_events[0]]
    for note in note_events[1:]:
        onset, offset, pitch, velocity, confidence = note
        prev_onset, prev_offset, prev_pitch, prev_velocity, prev_confidence = merge[-1]
        
        if (abs(onset - prev_onset) < min and pitch == prev_pitch):
            if velocity > prev_velocity:
                merge[-1] = note
        else:
            merge.append(note)
    return merge
    
        
        
    
def octave_error_correction(note_events, range=(40, 84)):
    """
    Ensures octave matches the constraints of the guitar range, setting them to the correct constraint.
    
    Parameters:
        note_events
        range: the guitar range of the octave of a guitar
    """
    corrected_octave = []
    min_midi, max_midi = range
    for note in note_events:
        onset, offset, pitch, velocity, confidence = note
        while (min_midi > pitch):
            pitch += 12
        while (max_midi < pitch):
            pitch -= 12
            
        corrected_octave.append((onset, offset, pitch, velocity, confidence))
    return corrected_octave
    
def sorted_note_time_pitch(note_events):
    """
    Sorts the note events based on onset time and then by pitch.
    x[0] = onset
    x[2] = pitch
    """
    key_func = lambda x: (x[0], x[2])
    
    sorted_tuples = sorted(note_events, key=key_func)
    
    return sorted_tuples

def smoothening_time(note_events, bpm=None, beat=None):
    """
    Smooths the onset time and offset to the closest beat using bpm and beat to quantize
    which grid to snap towards.
    
    Perimeters:
        bpm: beats per minute
        beat: Note type (quater 1/4, etc.)    
    """
    if beat == None and bpm == None:
        return note_events
    beat_dur_second = 60 / bpm
    quantize = beat_dur_second / beat
    smooth = []
    for note in note_events:
        onset, offset, pitch, velocity, confidence = note
        # quantizing to the nearest grid
        quantized_onset = round(onset / quantize) * quantize
        duration = offset - onset
        quantized_offset = quantized_onset + duration
        smooth.append((quantized_onset, quantized_offset, pitch, velocity, confidence))
    return smooth
        
    
    
def group_notes(note_events, epsilon=0.08, bpm=120, beats_bar=4, subdivisions=16):
    """
    Groups not into 16th bars 
    """
    new_note_event = []
    
    seconds_per_beat = 60.0 / bpm
    seconds_per_bar = seconds_per_beat * beats_bar
    seconds_per_slot = seconds_per_bar / subdivisions
    
    print(f"BPM: {bpm:.1f} | Bar: {seconds_per_bar:.3f}s | Slot (16th note): {seconds_per_slot:.3f}s")
    
    max_time = max(note_data[0] for note_data, _ in note_events)
    total_slots = int(max_time / seconds_per_slot) + 1
    
    slots = [[] for _ in range(total_slots)]
    for item in note_events:
        note_data, position = item
        onset = note_data[0]
        slot_index = int(onset / seconds_per_slot)
        if slot_index >= total_slots:
            continue
        string_num = position[0]
        while slot_index < total_slots:
            if not any(n[1][0] == string_num for n in slots[slot_index]):
                break
            slot_index += 1
        if slot_index < total_slots:
            slots[slot_index].append(item)
    
    return [slot for slot in slots if slot]

def detecting_removing_harmonics(note_events):
    """
    Remove harmonics that are being mistakenly detected as separate notes.
    https://www.coursera.org/learn/audio-signal-processing/lecture/dKdt9/harmonic-model
    
    """
    pass

def filter_process(note_events, scale_notes=None):
    """
    Creates a process pipeline which runs all the filtering processes
    
    Returns:
        A cleaned, filtered note_events
    """
    # 1: Filter by confidence
    notes = confidence_filter(note_events, 0.5)
    print(f"Number of notes after filtering by confience: {len(notes)} notes")
    
    # 2: Filter by short notes
    notes = short_note_filter(notes, 0.05)
    print(f"Number of notes after filtering by short notes: {len(notes)} notes")
    
    # : Sort the notes and pitch
    notes = sorted_note_time_pitch(notes)
    
    # 3: Removing duplicate notes
    notes = duplicate_note_filter(notes, 0.05)
    print(f"Number of notes after filtering by removing duplicate notes: {len(notes)} notes")
    
    # 4: Correcting octave to correct constraint
    notes = octave_error_correction(notes)
    
    return notes
    

    
    
