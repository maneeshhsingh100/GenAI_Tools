# speech_model_lib/asr.py
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import librosa


class ASRModel:
    def __init__(self, model_name='facebook/wav2vec2-large-960h'):
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

    def transcribe(self, audio_path):
        # Load audio
        speech, rate = librosa.load(audio_path, sr=16000)

        # Tokenize
        input_values = self.tokenizer(speech, return_tensors="pt").input_values

        # Perform inference
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Decode the logits to get the transcription
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.tokenizer.batch_decode(predicted_ids)[0]

        return transcription
