from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, conlist
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

model = joblib.load('../models/lightgbm_model.pkl')
with open('../models/best_threshold.txt', 'r') as f:
    best_threshold = float(f.read())

X_test = pd.read_csv("../../data/X_test.csv")
y_test = pd.read_csv("../../data/y_test.csv")

app = FastAPI(
    title="Fraud detection API",
    version="1.0.0"
)

#Data model
class TransactionFeatures(BaseModel):
    features: conlist(float, min_length=30, max_length=30)

class PredictionResult(BaseModel):
    prediction: int
    probability: float
    is_fraud: bool
    threshold: float
    message: str

@app.get("/")
def greet():
    return {"message": "bonjour"}

@app.post("/predict", response_model=PredictionResult, status_code=status.HTTP_200_OK)
def predict(transaction: TransactionFeatures):
    try:
        # On convertit les features en np.array
        features = np.array(transaction.features).reshape(1, -1)

        df = pd.DataFrame(features, columns=[
            "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
            "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
            "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
        ])

        if features.shape[1] != 30:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='30 features attendues'
            )

        prediction = model.predict(df)
        proba = model.predict_proba(df)

        # On détermine une fraude par rapport au seuil optimal
        is_fraud = proba[0][1] > best_threshold - 1e-6

        print(proba[0][1], best_threshold)

        return PredictionResult(
            prediction=int(prediction[0]),
            probability=float(proba[0][1]),
            is_fraud=bool(is_fraud),
            threshold=best_threshold,
            message="Fraude détectée" if is_fraud else "Transaction normale"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server error"
        )

@app.get("/metrics")
def get_metrics():
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        return metrics

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server error"
        )