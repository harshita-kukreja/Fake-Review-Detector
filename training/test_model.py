import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Example reviews
reviews = [
    "The hotel was amazing, I had a wonderful stay with great service.",
    "Worst experience ever, the staff was rude and the room was dirty.",
    "This place is fantastic, I highly recommend it to everyone!",
    "I think this review is fake, nothing about it seems real."
]

# Clean text like before
import re
def clean_review(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Predict
for r in reviews:
    clean = clean_review(r)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    label = "Deceptive (Fake)" if pred == 1 else "Truthful"
    print(f"Review: {r}\nPrediction: {label}\n")
