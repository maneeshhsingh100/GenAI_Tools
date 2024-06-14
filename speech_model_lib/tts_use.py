# Example usage script
from speech_model_lib import TTSModel

# Initialize TTS model
tts_model = TTSModel()

# Option 1: Synthesize speech from direct text input
text = "Hello, world!"
audio = tts_model.synthesize(text)
tts_model.save_wav(audio, sample_rate=22050, filepath='output_text.wav')

# Option 2: Synthesize speech from a text file
with open('input.txt', 'w') as file:
    file.write("This is a test from a file.")
audio = tts_model.synthesize('input.txt')
tts_model.save_wav(audio, sample_rate=22050, filepath='output_file.wav')
