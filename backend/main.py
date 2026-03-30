from pathlib import Path
import argparse
import soundfile as sf
import os
import sys
import torch
import numpy as np
import librosa
from backend.core.audio_loader import audio_loader
from backend.visualisation.plots import *
from backend.core.pitch_detector import *
from backend.tablature.fretboard_mapper import *
from backend.models import fretboard_mapper
from backend.tablature.tab_generator import *
from backend.analysis.note_filters import *
from backend.analysis.audio_seperation import *
from backend.tablature.fretboard_mapper import all_midi_notes


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from backend.models.goat.goat_cnn import GoatFretboardCNN
    from backend.models.goat.goat_prediction import get_goat_predictions
except ImportError:
    sys.path.append(os.path.join(project_root, "backend", "models", "goat"))

MODEL_PATH = os.path.join(project_root, "backend", "models", "models", "goat_3.pth")
STANDARD_TUNING = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}


def automatic_music_transcription(file_path, tuning, model_choice, use_demucs=False, min_confidence=0.5, min_duration=0.05, output_dir="outputs"):
    """
    Main function to handle transcription

    Perimeters:
        file_path: where the file is stored
        use_demucs: if True, run Demucs source separation before transcription
        min_confidence: the minimum accepted confidence of audio
        min_duration: the minimum accepted duration of note
        output_dir: position of output file
    """
    try:
        audio_path = Path(file_path)
        if (not audio_path.exists()):
            return f"error: audio file not found: {audio_path}"

        if use_demucs:
            print("Running Demucs source separation...")
            demucs_model = "mdx_extra_q"
            guitar_isolation(str(audio_path), demucs_model)
            separated_path = get_guitar_audio(str(audio_path), demucs_model)
            if not os.path.exists(separated_path):
                return f"error: Demucs output not found at {separated_path}"
            audio_path = Path(separated_path)
            print(f"Using separated guitar track: {audio_path}")

        audio_signal, sample_rate = audio_loader(audio_path)

        bpm, _ = librosa.beat.beat_track(y=audio_signal, sr=sample_rate)
        bpm = float(bpm)
        print(f"Current BPM: {bpm}")

        # Pre-processing
        # audio_signal, sample_rate = pre_process(audio_signal, sample_rate)

        # Converts back into wav file for Basic Pitch
        sf.write("processed.wav", audio_signal, sample_rate)
        file_path = "processed.wav"


        mapped_notes = []
        if model_choice == "End-to-End Architecture (GOAT Dataset)":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = GoatFretboardCNN().to(device)
            if os.path.exists(MODEL_PATH):
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                model.eval()
            else:
                return f"error: Model not found at {MODEL_PATH}"
            mapped_notes = get_goat_predictions(file_path, model, device)

        elif model_choice == "Heuristic (Baseline)":
            # Basic pitch detection ~ Spotify
            model_output, midi_data, note_events = pitch_detection(file_path)
            if not note_events:
                return "error: No notes detected in audio"
            # Processes the note events
            filtered_notes = filter_process(note_events)

            # Code for running FretBoardCNN
            # list of midi
            print("Sample note:", filtered_notes[0])

            mapped_notes = all_midi_notes(filtered_notes, STANDARD_TUNING)

            # Create flat list of (note_event, position) tuples


        elif model_choice == "Sequential Architecture (SynthDataset)":
            # Basic pitch detection ~ Spotify
            model_output, midi_data, note_events = pitch_detection(file_path)
            if not note_events:
                return "error: No notes detected in audio"
            # Processes the note events
            filtered_notes = filter_process(note_events)

            # Code for running FretBoardCNN
            # list of midi
            mapper = fretboard_mapper.FretBoardMapper()
            midi_list = [note[2] for note in filtered_notes]
            mapped = mapper.map_notes(midi_list)

            # Create flat list of (note_event, position) tuples
            mapped_notes = []
            for note_event, position in zip(filtered_notes, mapped):
                mapped_notes.append((note_event, position))
        else:
            return f"error: Invalid model choice selected: {model_choice}"


        print(f"Created {len(mapped_notes)} mapped notes")


        # sorted_notes = sorted_notes(mapped_notes)
        grouped_notes = group_notes(mapped_notes, bpm=bpm)

        # save_midi(audio_path)

        full_tab = Tab.display_ascii_tab(grouped_notes, time_signature=4, subdivisions=16)
        return full_tab, mapped_notes, bpm
    except Exception as e:
        return str(e)

def main():
    """
    Command line interface
    """
    parser = argparse.ArgumentParser(description='CLI for Guitar AMT')
    parser.add_argument("file_path", help="Path to the audio file (.wav or .mp3)")
    args = parser.parse_args()

    MODELS = {
        "1": "Sequential Architecture (SynthDataset)",
        "2": "End-to-End Architecture (GOAT Dataset)",
        "3": "Heuristic (Baseline)",
    }
    print("\nSelect model:")
    for key, name in MODELS.items():
        print(f"  {key}) {name}")
    model_choice_input = input("Enter number (1/2/3): ").strip()
    model_choice = MODELS.get(model_choice_input)
    if not model_choice:
        print(f"Invalid choice '{model_choice_input}'. Defaulting to End-to-End Architecture.")
        model_choice = MODELS["2"]

    use_demucs_input = input("\nRun Demucs source separation first? (Y/N): ").strip().upper()
    use_demucs = use_demucs_input == "Y"

    print(f"\nRunning transcription with: {model_choice}")
    print(f"Demucs: {'ON' if use_demucs else 'OFF'}\n")

    result = automatic_music_transcription(args.file_path, None, model_choice, use_demucs=use_demucs)

    if isinstance(result, tuple):
        full_tab, mapped_notes, bpm = result
        print(f"Detected BPM: {bpm:.1f}")
        print(f"Notes mapped: {len(mapped_notes)}")
        print("\n" + full_tab)
    else:
        print(result)


if __name__ == '__main__':
    main()
