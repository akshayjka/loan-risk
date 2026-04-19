from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import os

from fastapi.middleware.cors import CORSMiddleware

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

app = FastAPI(title="AI Loan Underwriting API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "loan_approval.csv")

# Load Model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model Loaded")
else:
    model = None
    print("❌ Model Not Found")


# ----------------------------
# Request Schema
# ----------------------------
class LoanApp(BaseModel):
    income: float
    loan_amount: float
    cibil_score: int = Field(..., alias="cibil_score")
    years_employed: float = 2.0


# ----------------------------
# Root
# ----------------------------
@app.get("/")
def root():
    return {
        "status": "Online",
        "model_loaded": model is not None
    }


# ----------------------------
# Predict API
# ----------------------------
@app.post("/predict")
def predict_loan(data: LoanApp):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_data = np.array([[
            data.income,
            data.loan_amount,
            data.cibil_score,
            data.years_employed
        ]])

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        return {
            "status": "Approved" if int(pred) == 1 else "Rejected",
            "probability": round(float(prob) * 100, 2),
            "engine": "XGBoost"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Model Quality API
# ----------------------------
@app.get("/metrics")
def model_metrics():

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        # Load dataset
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.strip()

        # Target
        df["loan_approved"] = df["loan_approved"].replace({
            "Yes": 1,
            "No": 0
        })

        y = df["loan_approved"]

        # Features
        X = df[[
            "income",
            "loan_amount",
            "credit_score",
            "years_employed"
        ]]

        # Same split as training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Feature Importance
        importance = model.feature_importances_
        feature_names = X.columns.tolist()

        feature_data = []
        for name, score in zip(feature_names, importance):
            feature_data.append({
                "name": name,
                "value": round(float(score) * 100, 2)
            })

        return {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),

            "confusion_matrix": {
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn)
            },

            "feature_importance": feature_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Run Server
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)