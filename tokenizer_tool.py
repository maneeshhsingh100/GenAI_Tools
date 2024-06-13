# tokenizer code

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

# Sample text data
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "This is a sample sentence for tokenization.",
    "Tokenization is an important NLP task."
]

# Create tokenizer object
tokenizer = Tokenizer()

# Fit tokenizer on text data
tokenizer.fit_on_texts(texts)

# Convert text to sequences of tokens
sequences = tokenizer.texts_to_sequences(texts)

# Vocabulary
word_index = tokenizer.word_index
print("Word index:", word_index)

# Tokenized sequences
print("Tokenized sequences:", sequences)