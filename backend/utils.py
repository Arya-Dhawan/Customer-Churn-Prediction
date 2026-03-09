import joblib
import shap
import numpy as np
import pandas as pd

model = joblib.load("../models/random_forest_tuned.pkl")
preprocessor = joblib.load("../models/preprocessor.pkl")
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