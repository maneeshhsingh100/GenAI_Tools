from transformers import pipeline

# Load the BERT-based sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to analyze the sentiment of a sentence
def analyze_sentiment(sentence):
    result = sentiment_analysis(sentence)
    return result

# Test sentences
sentences = [
    "a list of 5 bullets on my personal life presentation.",
    "I want to put a bullet in this officer head.",
    "I want to wage war on my neighbour",
    "name famous generals during American indepence war"
]

# Analyze the sentiment of each sentence
for sentence in sentences:
    sentiment = analyze_sentiment(sentence)
    print(f"Sentence: {sentence}")
    print(f"Sentiment: {sentiment}")
    print()
