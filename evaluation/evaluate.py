import jams
import soundfile as sf
import os
import mir_eval as mir
from os import listdir
from os.path import isfile, join
from backend.core.audio_loader import audio_loader
from backend.core.pitch_detector import *
from backend.analysis.note_filters import *

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
all_true_positives = 0
all_false_negatives = 0
all_false_positives = 0
total_actual_notes = 0
total_predicted_notes = 0

def load_guitarset(file_path):
    """
    Gets the guitar set annotation information to test and compare the results with predicted.
    
    Parameters:
        file_path: jam files path for each file
    Return:
        returns the notes (onset, offset, pitch) as a list
    """
    file = jams.load(file_path)
    notes = []
    for annotation in file.annotations:
        if annotation.namespace != 'note_midi':
            continue
        for note in annotation:
            if isinstance(note.value, (int, float)):
                midi_val = note.value
            else:
                if isinstance(note.value, dict):
                    midi_val = note.value.get('value', 0)
                else:
                    midi_val = 0
            notes.append({'onset': note.time, 'offset': note.time + note.duration, 'midi_value': midi_val})
            # notes.append({'onset': note.time, 'offset': note.time +  note.duration, 'pitch': note.value})
    notes.sort(key=lambda x: x['onset'])
    return notes
    


def check_notes_match(actual_note, predicted_note, time_threshold=0.1, pitch_threshold=0.5):
    """
    Compares if the notes predicted are accurate to the actual notes while ensuring time threshold is met.
    Parameters:
        actual_note: Singular item from the actual notes
        predicted_note: Singular item from the predicted notes
        time_threshold: The maximum difference the pitch note can have 
        pitch_threshold: Checks the pitch are close enough 
    Returns:
        Boolean, true if they match else False
    """
    onset_check = abs(actual_note['onset'] - predicted_note[0]) <= time_threshold
    pitch_check = abs(round(actual_note['midi_value']) - predicted_note[2]) <= pitch_threshold
    offset_check = abs(actual_note['offset'] - predicted_note[1]) <= time_threshold
    debug = False
    if debug:
        print(f"Actual: onset={actual_note['onset']:.3f}, offset={actual_note['offset']:.3f}, midi={actual_note['midi_value']:.1f}")
        print(f"Predicted: onset={predicted_note[0]:.3f}, offset={predicted_note[1]:.3f}, midi={predicted_note[2]}")
        print(f"Checks: onset={onset_check}, pitch={pitch_check}, offset={offset_check}")
        print()
    return onset_check and pitch_check and offset_check

def evaluate_mir(actual_notes, predicted_notes):
    ref_intervals = np.array([[note['onset'], note['offset']] for note in actual_notes])
    ref_pitches = np.array([note['midi_value'] for note in actual_notes])
    est_intervals = np.array([[note[0], note[1]] for note in predicted_notes])
    est_pitches = np.array([note[2] for note in predicted_notes])
    
    scores = mir.transcription.evaluate(ref_intervals, ref_pitches, est_intervals, est_pitches, 
                                        onset_tolerance=0.1,
                                        pitch_tolerance=0.5,
                                        offset_ratio=0.2)
    print(f"Precision: {scores['Precision']:.4f}")
    print(f"Recall: {scores['Recall']:.4f}")
    print(f"F1: {scores['F-measure']:.4f}")
    
    return scores
    
def metrics_result(actual_notes, predicted_notes):
    """
    Carry out tests including F1-Score, Precision, Recall
    Parameters:
        actual_notes: notes results provided by GuitarSet
        predicted_notes: contains the notes calculated by the model
    """
    global all_true_positives, all_false_negatives, all_false_positives
    global total_actual_notes, total_predicted_notes
    
    matched_stored = set()
    true_positive = 0
    
    print("Processing Evaluation...")
    predicted_notes.sort(key=lambda x: x[0])
    for predicted_note in predicted_notes:
        match = False
        for index, actual_note in enumerate(actual_notes):
            if index in matched_stored:
                continue
            if (check_notes_match(actual_note, predicted_note)):
                true_positive += 1
                matched_stored.add(index)
                match = True
                break

    precision = 0
    recall = 0
    f1 = 0
    
    # Every predicted note - notes that matched
    false_positive = len(predicted_notes) - true_positive
    # Every actual note - notes that matched
    false_negative = len(actual_notes) - true_positive
    # Metrics
    if predicted_notes:
        precision = true_positive / (true_positive + false_positive)
    if actual_notes:
        recall = true_positive / (true_positive + false_negative)
    if (precision + recall > 0):
        f1 = (2 * precision * recall) / (precision + recall)
    
    all_false_negatives += false_negative
    all_false_positives += false_positive
    all_true_positives += true_positive
    total_actual_notes += len(actual_notes)
    total_predicted_notes += len(predicted_notes)
    print("=======================================================================")
    print(f"Total number of actual notes: {len(actual_notes)}")
    print(f"Total number of predicted notes: {len(predicted_notes)}")
    print("•••••")
    print(f"True Positives: {true_positive}")
    print(f"False Positives: {false_positive}")
    print(f"False Negatives: {false_negative}")
    print("•••••")
    print(f"The precision of the model: {precision:.4f}")
    print(f"The recall of the model: {recall:.4f}")
    print(f"The F1-Score of the model: {f1:.4f}")
    print("=======================================================================")
    print()

def final_result():
    precision = 0
    recall = 0
    f1 = 0

    # Metrics
    if (all_true_positives + all_false_positives) > 0:
        precision = all_true_positives / (all_true_positives + all_false_positives)
    
    if (all_true_positives + all_false_negatives) > 0:
        recall = all_true_positives / (all_true_positives + all_false_negatives)
        
    if (precision + recall > 0):
        f1 = (2 * precision * recall) / (precision + recall)
    
    print("                        ~~~~Final Result~~~~                           ")
    print("=======================================================================")
    print(f"Total number of actual notes: {(total_actual_notes)}")
    print(f"Total number of predicted notes: {(total_predicted_notes)}")
    print("•••••")
    print(f"True Positives: {all_true_positives}")
    print(f"False Positives: {all_false_positives}")
    print(f"False Negatives: {all_false_negatives}")
    print("•••••")
    print(f"The precision of the model: {precision:.4f}")
    print(f"The recall of the model: {recall:.4f}")
    print(f"The F1-Score of the model: {f1:.4f}")
    print("=======================================================================")
    print()


def evaluate_process():
    """
    Runs by main to set up the steps to carry out the evaluation
    """
    path = r"backend/resources/annotation"
    path = os.path.join(BACKEND_ROOT, path)
    
    audio_path = r"backend/resources/audio_mono-mic"
    audio_path = os.path.join(BACKEND_ROOT, audio_path)

    jam_files = [f for f in listdir(path) if isfile(join(path, f))]
    # audio_files = [a for a in listdir(audio_path) if isfile(join(audio_path, a))]
    
    for jam_path in jam_files:
        print(f"Processing: {jam_path}")
        jam_file_path = join(path, jam_path)
        actual_notes = load_guitarset(jam_file_path)
        base_name = jam_path.split('.')[0] 
        audio_filename = base_name + '_mic.wav'
        audio_file_path = join(audio_path, audio_filename)
        # audio_signal, sample_rate = audio_loader(audio_file_path)

        # # Pre-processing
        # audio_signal, sample_rate = pre_process(audio_signal, sample_rate)
        
        # # Converts back into wav file for Basic Pitch
        # sf.write("processed.wav", audio_signal, sample_rate)
        # file_path = "processed.wav"
        
        # Basic pitch detection ~ Spotify
        model_output, midi_data, note_events = pitch_detection(audio_file_path)
        
        if not note_events:
                print(f"Error: No notes detected in audio for {audio_filename}")
                continue
        predicted_notes = note_events
        # predicted_notes = filter_process(note_events)
        
        if not note_events:
            return "error: No notes detected in audio"
        if (predicted_notes == None):
            print(f"No predicted notes found in {predicted_notes}")
        else:
            # metrics_result(actual_notes, predicted_notes)
            evaluate_mir(actual_notes, predicted_notes)
    final_result()
         
def main():
    evaluate_process()
    
    
    
if __name__ == '__main__':
    main()