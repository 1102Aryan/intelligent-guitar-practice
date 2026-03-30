import librosa
import matplotlib.pyplot as plt
import numpy as np

def onset_detection(audio, sample_rate):
    onset_time = librosa.onset.onset_detect(y=audio, sr=sample_rate, units='time')
    print(onset_time)
    return onset_time

def onset_graph(audio, onset_time, sample_rate):
    # Creates an onset graph to show the start time of audio.
    librosa.display.waveshow(audio, sr=sample_rate, alpha=0.5)
    plt.vlines(onset_time, -1, 1, color='g', linestyle='dashed', label='Onsets')
    plt.legend()
    plt.title('Onset Detection')
    plt.tight_layout()
    plt.show()
    
    
def spectogram(audio):
    # Creates a spectogram of the audio for representation.
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(S_db,
                                x_axis='time',
                                y_axis='log',
                                ax=ax)
    ax.set_title('Spectogram Example', fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    plt.show()