import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# -----------------------
# 1. Load Data
# -----------------------
df = pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\fake review detector\deceptive-opinion.csv")

# Map labels: truthful -> 0, deceptive -> 1
df['label'] = df['deceptive'].map({'truthful': 0, 'deceptive': 1})

# -----------------------
# 2. Text Cleaning
# -----------------------
def clean_review(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # keep letters & numbers
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_review)

# -----------------------
# 3. Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

print("Train distribution:\n", y_train.value_counts())
print("Test distribution:\n", y_test.value_counts())

# -----------------------
# 4. Vectorize
# -----------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))  # unigrams + bigrams
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------
# 5. Train Model (LinearSVC)
# -----------------------
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# -----------------------
# 6. Evaluate
# -----------------------
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# -----------------------
# 7. Save Model + Vectorizer
# -----------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully with LinearSVC!")
