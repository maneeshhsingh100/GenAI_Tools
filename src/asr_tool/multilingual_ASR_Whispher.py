import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from langdetect import detect

# Load and preprocess audio
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

# Transcribe audio
def transcribe_audio(audio, sr, model, processor):
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

# Load ASR model and processor
model_name = "openai/whisper-large-v2"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Process multiple files
file_paths = ["file1.wav", "file2.wav"]  # List of your .wav files
for file_path in file_paths:
    audio, sr = load_audio(file_path)
    transcription = transcribe_audio(audio, sr, model, processor)
    language = detect(transcription)
    print(f"File: {file_path}")
    print("Detected Language:", language)
    print("Transcription:", transcription)
    print("-" * 50)
