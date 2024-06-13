# Install the necessary libraries
#pip install transformers

from transformers import pipeline

# Load the text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Function to generate sentences
def generate_sentences(word, sentiment, n=50):
    sentences = []
    prompt = f"Write a {sentiment} sentence using the word '{word}':"
    for _ in range(n):
        result = generator(prompt, max_length=30, num_return_sequences=1)
        sentences.append(result[0]['generated_text'].strip())
    return sentences

def generate_for_word(word):
    positive_sentences = generate_sentences(word, 'positive')
    negative_sentences = generate_sentences(word, 'negative')
    return positive_sentences, negative_sentences

# Load the word list from the uploaded file
word_list_path = '/path_your/wordlist.txt'

with open(word_list_path, 'r') as file:
    words = file.read().splitlines()

# Remove empty lines if any
words = [word for word in words if word.strip()]

# Generate sentences for each word and save them to a dictionary
sentences_dict = {}
for word in words:
    print(f"Generating sentences for: {word}")
    positive_sentences, negative_sentences = generate_for_word(word)
    sentences_dict[word] = {
        "positive": positive_sentences,
        "negative": negative_sentences
    }

# Save the generated sentences to a file
import json

output_file_path = '/output_path/generated_sentences.txt'

with open(output_file_path, 'w') as outfile:
    json.dump(sentences_dict, outfile)

print(f"Sentences saved to {output_file_path}")

# Provide a link to download the generated sentences file
from google.colab import files

files.download(output_file_path)