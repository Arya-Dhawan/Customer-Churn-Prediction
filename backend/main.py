from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import matplotlib
import pandas as pd
import base64
import io
import matplotlib.pyplot as plt
import shap

from schemas import CustomerData
from feature_engineering import engineer_features
from utils import predict_single, predict_batch, shap_explain

app = FastAPI(title="Customer Churn Prediction API")

# --------------------------
# Health Check
# --------------------------
@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}

# --------------------------
# Single Prediction
# --------------------------
@app.post("/predict")
def predict(data: CustomerData):

    df = pd.DataFrame([data.model_dump()])
    df = engineer_features(df)

    prob = predict_single(df)

    risk = (
        "High" if prob > 0.7 else
        "Medium" if prob > 0.4 else
        "Low"
    )

    return {
        "churn_probability": float(prob),
        "risk_level": risk
    }

# --------------------------
# Batch Prediction
# --------------------------
@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)
    df = engineer_features(df)

    probs = predict_batch(df)

    df["churn_probability"] = probs
    df["risk_level"] = df["churn_probability"].apply(
        lambda x: "High" if x > 0.7 else "Medium" if x > 0.4 else "Low"
    )

    return df.to_dict(orient="records")

# --------------------------
# SHAP Plot (Waterfall)
# --------------------------
matplotlib.use("Agg")
@app.post("/shap")
def shap_plot(data: CustomerData):
    try:
        df = pd.DataFrame([data.model_dump()])
        df = engineer_features(df)

        shap_values, processed, feature_names = shap_explain(df)

        plt.figure()
        shap.plots.waterfall(shap_values[0,:,0], show=False)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)

        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return {"image": img_base64}
    except Exception as e:
        return {"error": str(e)}
# --------------------------
# Model Insights (static)
# --------------------------
@app.get("/insights")
def insights():
    return {
        "roc_curve": "../models/ROC_curve.png",
        "confusion_matrix": "../models/confusion_matrix.png",
        "feature_importance": "../models/feature_importance.png"
    }