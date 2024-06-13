import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample training data
texts = [
    "John lives in New York City.",
    "Google is headquartered in Mountain View.",
    "Alice works at Microsoft in Seattle."
]

# Corresponding entity labels
labels = [
    [(0, 4, 'PERSON'), (13, 15, 'LOCATION')],
    [(0, 6, 'ORGANIZATION'), (26, 38, 'LOCATION')],
    [(0, 5, 'PERSON'), (15, 24, 'ORGANIZATION'), (28, 35, 'LOCATION')]
]

# Create vocabulary and label dictionaries
word_to_idx = {}
label_to_idx = {'PAD': 0}
for text, label_list in zip(texts, labels):
    for word in text.split():
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx) + 1
    for label in label_list:
        if label[2] not in label_to_idx:
            label_to_idx[label[2]] = len(label_to_idx)

# Convert text and labels to sequences of indices
text_sequences = [[word_to_idx[word] for word in text.split()] for text in texts]

# Pad text sequences to ensure uniform length
max_length = max(len(seq) for seq in text_sequences)
padded_text_sequences = pad_sequences(text_sequences, maxlen=max_length, padding='post')

# Create padded label sequences
padded_label_sequences = []
for label_list in labels:
    label_seq = np.zeros(max_length)
    for label in label_list:
        start, end, entity = label
        label_seq[start:end] = label_to_idx[entity]
    padded_label_sequences.append(label_seq)

# Convert to numpy arrays
padded_label_sequences = np.array(padded_label_sequences)

# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_to_idx) + 1, 32, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Dense(len(label_to_idx), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_text_sequences, padded_label_sequences, epochs=10, verbose=1)

# Example testing
test_text = "Google is headquartered in Mountain View."
test_sequence = [word_to_idx[word] if word in word_to_idx else 0 for word in test_text.split()]
padded_test_sequence = pad_sequences([test_sequence], maxlen=max_length, padding='post')
prediction = model.predict(padded_test_sequence)[0]
predicted_labels = [np.argmax(pred) for pred in prediction]

# Map predicted label indices to their corresponding entity labels
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
predicted_entities = [idx_to_label[idx] for idx in predicted_labels]

print("Test Text:", test_text)
print("Predicted Entities:", predicted_entities)