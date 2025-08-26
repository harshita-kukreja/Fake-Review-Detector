**Fake Review Detector**
Fake Review Detector is an AI-powered application that identifies deceptive hotel reviews.
Using NLP techniques and machine learning models (TF-IDF + Logistic Regression / Linear SVC), this project helps distinguish between truthful and fake reviews.

**Dataset**
I used the Deceptive Opinion Spam Corpus:
400 truthful positive reviews (TripAdvisor)
400 deceptive positive reviews (Mechanical Turk)
400 truthful negative reviews (Expedia, Hotels.com, etc.)
400 deceptive negative reviews (Mechanical Turk)
Total: 1600 hotel reviews (balanced dataset).

**Installation**
Clone the repository and install dependencies:
git clone https://github.com/harshita-kukreja/Fake-Review-Detector.git
cd Fake-Review-Detector
pip install -r requirements.txt

**Training the Model**
cd training
python train.py
This saves the model and vectorizer in the project folder.

**Running the App**
Start the FastAPI server:
uvicorn app:app --reload

Open your browser at: http://127.0.0.1:8000/docs
Here you can test the /predict endpoint by entering any review text.

**Future Improvements:**
Deploy on a cloud platform (Heroku, Render, or AWS).
Experiment with transformer models like BERT for higher accuracy.
Extend dataset to include Indian hotel reviews for localized predictions.
