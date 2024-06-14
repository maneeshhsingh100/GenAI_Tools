# Example usage script
from speech_model_lib import ASRModel

# Initialize ASR model
asr_model = ASRModel()

# Transcribe audio file
transcription = asr_model.transcribe('path/to/audio.wav')
print(f'Transcription: {transcription}')
