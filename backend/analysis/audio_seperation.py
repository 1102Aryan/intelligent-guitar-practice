import os
import demucs.separate

def guitar_isolation(audio_file, model="mdx_extra_q"):
    """
    Demucs is an open source audio isolation tool made by Facebook's research team.
    model (mdx_extra longer wait better quality, mdx_extra_q faster model, quality is worser)
    Creates an ouput of the audio file in separated/{model}/{song name}
    """
    demucs.separate.main(["-n", model, audio_file])
    
    
def get_guitar_audio(audio_file, model):
    """
    Gets the audio file (other.wav: contains guitar track) for processing and tab generation.
    Perimeter:
        audio_file: gets the file name of the audio file
        model: gets the model name to find the guitar audio path
    Returns:
        isolated audio path
    """
    name = os.path.splitext(os.path.basename(audio_file))[0]
    folder = f"separated/{model}/{name}"
    guitar_path = os.path.join(folder, "other.wav")
    return guitar_path