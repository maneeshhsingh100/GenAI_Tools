import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

# Sample data (replace with your own dataset)
texts = [
    "I love this movie!",
    "This movie is great.",
    "The film is fantastic.",
    "I didn't like the acting.",
    "The plot was boring."
]
labels = [1, 1, 1, 0, 0]  # 1 for positive, 0 for negative

# Tokenize the text data
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to ensure uniform length
max_len = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Convert labels to numpy array
labels = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, 16, input_length=max_len),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Example test on new data
new_texts = [
    "This movie was terrible.",
    "I enjoyed the film a lot."
]
new_sequences = tokenizer.texts_to_sequences(new_texts)
padded_new_sequences = pad_sequences(new_sequences, maxlen=max_len)
predictions = model.predict(padded_new_sequences)
for i, text in enumerate(new_texts):
    print("Text:", text)
    print("Predicted Sentiment:", "Positive" if predictions[i] > 0.5 else "Negative")