import nltk
from nltk.corpus import wordnet
import random


def rephrase_sentence(sentence, replace_prob=0.5):
  """
  Rephrases a sentence by substituting synonyms for nouns and verbs with some randomness.

  Args:
      sentence: The sentence to rephrase (string)
      replace_prob: The probability of replacing a word with a synonym (float)

  Returns:
      A list of rephrased sentences (list of strings)
  """
  # Download necessary resources from NLTK
  nltk.download('punkt')
  nltk.download('wordnet')

  # Tokenize the sentence
  tokens = nltk.word_tokenize(sentence)

  # Generate rephrased versions
  rephrased_sentences = []
  for i, token in enumerate(tokens):
    if random.random() < replace_prob and nltk.pos_tag([token])[0][1] in ['NN', 'VB']:  # Check for nouns (NN) and verbs (VB) with probability
      synonyms = wordnet.synsets(token)
      if synonyms:
        # Choose a random synonym from the list
        new_token = random.choice(synonyms).lemmas()[0].name()
        new_sentence = " ".join([tokens[j] if j != i else new_token for j in range(len(tokens))])
        rephrased_sentences.append(new_sentence)

  # Add the original sentence to the list
  rephrased_sentences.append(sentence)
  return rephrased_sentences


# Example usage
def process_line_by_line(input_file, output_file):
  """
  Reads an input file line by line, prepends "Who achieved the" to each line,
  and writes the modified lines to a new output file.

  Args:
      input_file: Path to the input text file (string)
      output_file: Path to the output text file (string)
  """
  with open(input_file, 'r') as f, open(output_file, 'w') as f_out:
    for line in f:
      new_line = f"Who achieved the {line.strip()}"  # Remove trailing newline with strip()
      f_out.write(new_line)

# ... (Rest of the code for process_text_file remains the same)

input_file = "training_2.txt"  # Replace with your input file path
output_file = "rephrased_sentences_random.txt"

print(f"Successfully processed {input_file} and wrote to {output_file}")
