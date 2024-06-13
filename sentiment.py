from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis")

# Function to analyze the sentiment of a sentence
def analyze_sentiment(sentence):
    result = sentiment_analysis(sentence)
    return result

# Test sentences
sentences = [
    "I love this product! It works great and is exactly what I needed.",
    "I am very disappointed with this service. It was terrible and I won't be using it again."
]

# Analyze the sentiment of each sentence
for sentence in sentences:
    sentiment = analyze_sentiment(sentence)
    print(f"Sentence: {sentence}")
    print(f"Sentiment: {sentiment}")
    print()
