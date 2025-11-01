from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pickle
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predictions: dict

# Load model at startup
class StockPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DistilBertForSequenceClassification.from_pretrained('model-export/model')
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('model-export/model')
        self.model.to(self.device)
        self.model.eval()
        
        with open('model-export/normalization_params.pkl', 'rb') as f:
            params = pickle.load(f)
            self.means = np.array([params['means'][s] for s in params['stocks']])
            self.stds = np.array([params['stds'][s] for s in params['stocks']])
            self.stocks = params['stocks']
    
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                               max_length=256, padding='max_length')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            preds = outputs.logits.cpu().numpy()
        
        preds_denorm = (preds * self.stds + self.means) * 100
        return {self.stocks[i]: float(preds_denorm[0, i]) for i in range(len(self.stocks))}

predictor = StockPredictor()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    predictions = predictor.predict(request.text)
    return {"predictions": predictions}

@app.get("/health")
async def health():
    return {"status": "ok"}