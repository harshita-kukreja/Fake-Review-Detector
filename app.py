from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib

# Load model + vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class Review(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(review: Review):
    X = vectorizer.transform([review.text])
    pred = model.predict(X)[0]
    label = "Truthful" if pred == 0 else "Deceptive (Fake)"
    return {"prediction": label}
