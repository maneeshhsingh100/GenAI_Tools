# speech_model_lib/tts.py
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
from .utils import read_text_from_file

# Load Tacotron 2 model and WaveGlow model
tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow')


class TTSModel:
    def __init__(self, tacotron2_model=tacotron2, waveglow_model=waveglow):
        self.tacotron2 = tacotron2_model
        self.waveglow = waveglow_model

    def synthesize(self, input_text_or_file):
        # Check if input is a text file path or a text string
        if os.path.isfile(input_text_or_file):
            text = read_text_from_file(input_text_or_file)
        else:
            text = input_text_or_file

        # Prepare input text
        sequence = np.array(self.tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.from_numpy(sequence).to(torch.int64).cuda()

        # Generate mel-spectrogram using Tacotron 2
        with torch.no_grad():
            _, mel, _, _ = self.tacotron2.infer(sequence)

        # Generate audio using WaveGlow
        with torch.no_grad():
            audio = self.waveglow.infer(mel)

        # Convert audio to CPU and remove batch dimension
        audio = audio[0].data.cpu().numpy()

        return audio

    def save_wav(self, audio, sample_rate, filepath):
        librosa.output.write_wav(filepath, audio, sample_rate)

    def plot_mel(self, mel):
        fig, ax = plt.subplots(figsize=(10, 2))
        im = ax.imshow(mel, aspect='auto', origin='lower')
        plt.colorbar(im, ax=ax)
        plt.show()
