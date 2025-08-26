from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import onnxruntime as rt 
# For ONNX model inference
# import onnxruntime as rt 

# --- 1. App Creation and Model Loading ---
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="An API to detect fraudulent credit card transactions in real-time.",
    version="1.0.0"
)

# Load the preprocessing pipeline
with open('processor/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Load the trained model (pickle version)
# This assumes the best model from MLflow was saved as 'best_model.pkl'
# with open('best_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# --- For ONNX model ---
sess = rt.InferenceSession("model/fraud_detector.onnx")
input_name = sess.get_inputs()[0].name

# --- 2. Pydantic Models for Input and Output ---
# Define the input data structure based on the features the model expects
# This should match the columns of X_train
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Hour: int
    DayOfWeek: int
    Avg_Amount: float
    Amount_to_Avg_Amount: float
    Trans_Freq: float

class PredictionResponse(BaseModel):
    is_fraud: int
    probability: float
    
# --- 3. API Endpoints ---
@app.get("/", tags=["Health Check"])
def read_root():
    """A simple health check endpoint."""
    return {"status": "API is running"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_fraud(transaction: Transaction):
    """
    Predicts if a transaction is fraudulent based on its features using ONNX model.
    """
    try:
        # Convert Pydantic model to DataFrame
        data = pd.DataFrame([transaction.dict()])

        # Preprocess
        preprocessed_data = preprocessor.transform(data).astype(np.float32)

        # Run ONNX session
        input_name = sess.get_inputs()[0].name
        pred_proba = sess.run(None, {input_name: preprocessed_data})[0]

        # Extract probability (fraud class = 1)
        probability = float(pred_proba[0][1])  
        prediction = 1 if probability > 0.5 else 0

        return {
            "is_fraud": prediction,
            "probability": probability
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example Input JSON for the /predict endpoint:
# {
#   "V1": -1.3598, "V2": -0.0727, "V3": 2.5363, "V4": 1.3781, "V5": -0.3383,
#   "V6": 0.4623, "V7": 0.2395, "V8": 0.0986, "V9": 0.3637, "V10": 0.0907,
#   "V11": -0.5516, "V12": -0.6178, "V13": -0.9913, "V14": -0.3111, "V15": 1.4681,
#   "V16": -0.4704, "V17": 0.2079, "V18": 0.0257, "V19": 0.4039, "V20": 0.2514,
#   "V21": -0.0183, "V22": 0.2778, "V23": -0.1104, "V24": 0.0669, "V25": 0.1285,
#   "V26": -0.1891, "V27": 0.1335, "V28": -0.0210, "Amount": 149.62,
#   "HourOfDay": 0, "DayOfWeek": 0, "Avg_Amount_Card_So_Far": 150.0,
#   "Amount_to_Avg_Amount": 0.997, "Time_Since_Last_Tx": 0.0
# }

# Example Output JSON:
# {
#   "is_fraud": 0,
#   "probability": 0.00123
# }