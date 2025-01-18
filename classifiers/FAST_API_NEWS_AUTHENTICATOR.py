from fastapi import FastAPI
import gensim.downloader as api
import spacy
import pickle
import numpy as np
from pydantic import BaseModel
import os

app = FastAPI()

# Load models once at startup
GENMOD = api.load("word2vec-google-news-300")
print("Word2Vec Google News model (GENMOD) loaded successfully.")
nlp = spacy.load("en_core_web_lg")
print("spaCy large English model (nlp) loaded successfully.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "New_True_Fake.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    knnm = pickle.load(f)

class NewsRequest(BaseModel):
    text: str

def preprocess_and_vectorize(text):
    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    if words:
        return GENMOD.get_mean_vector(words).reshape(1, -1) 
    else:
        return np.zeros((1, 300))

@app.post("/predict/")
async def predict(news: NewsRequest):
    vector = preprocess_and_vectorize(news.text)
    prediction = knnm.predict(vector)
    return {"prediction": int(prediction[0])}
