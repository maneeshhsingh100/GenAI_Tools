# word embedding code

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text data
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "This is a sample sentence for word embedding.",
    "Word embedding is a useful technique in NLP."
]

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

# Define embedding dimension
embedding_dim = 100

# Create embedding matrix
embedding_matrix = tf.random.uniform((vocab_size, embedding_dim))

# Create embedding layer
embedding_layer = tf.keras.layers.Embedding(
    vocab_size,
    embedding_dim,
    weights=[embedding_matrix],
    input_length=max_length,
    trainable=False  # Set to True if you want to fine-tune embeddings
)

# Example model using embedding layer
model = tf.keras.Sequential([
    embedding_layer,
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Display model summary
print(model.summary())

# Train the model (example)
# Replace X_train and y_train with your actual data
# model.fit(padded_sequences, y_train, epochs=10, validation_split=0.2)
