import joblib
import shap
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model_path = os.path.join(BASE_DIR, "models", "random_forest_tuned.pkl")
prep_path  = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
model = joblib.load(model_path)
preprocessor = joblib.load(prep_path)
explainer = shap.TreeExplainer(model)

# --------------------------
# Single Prediction
# --------------------------
def predict_single(df):
    processed = preprocessor.transform(df)
    prob = model.predict_proba(processed)[0,1]
    return prob

# --------------------------
# Batch Prediction
# --------------------------
def predict_batch(df):
    processed = preprocessor.transform(df)
    probs = model.predict_proba(processed)[:,1]
    return probs

# --------------------------
# SHAP Explanation
# --------------------------
def shap_explain(df):
    processed = preprocessor.transform(df)
    feature_names = preprocessor.get_feature_names_out()
    processed_df = pd.DataFrame(processed, columns=feature_names)
    shap_values = explainer(processed_df)
    return shap_values, processed_df, feature_names