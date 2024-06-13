# text classification code

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text data and corresponding labels
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "This is a sample sentence for text classification.",
    "Text classification is an important NLP task."
]
labels = [0, 1, 1]  # Example labels (0 for negative, 1 for positive)

# Convert labels to numpy array
labels = np.array(labels)

# Create tokenizer object
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Convert text to sequences of tokens
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to ensure uniform length
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Vocabulary
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=10, verbose=1)

# Example prediction
test_text = ["The lazy cat sleeps."]
test_sequences = tokenizer.texts_to_sequences(test_text)
test_padded_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post')
predictions = model.predict(test_padded_sequences)
print("Prediction:", predictions)