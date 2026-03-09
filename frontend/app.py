import streamlit as st
import requests
import pandas as pd
import base64

API = "https://customer-churn-prediction-ct21.onrender.com"

st.set_page_config(page_title="Churn Intelligence Dashboard", layout="wide")

st.title("📊 Customer Churn Intelligence Dashboard")

page = st.sidebar.selectbox(
    "Navigation",
    ["Customer Prediction", "Batch Upload", "Model Insights"]
)

# =========================================================
# PAGE 1 — CUSTOMER PREDICTION
# =========================================================
if page == "Customer Prediction":

    st.header("Individual Customer Prediction")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0,1])
        Partner = st.selectbox("Partner", ["Yes","No"])
        Dependents = st.selectbox("Dependents", ["Yes","No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        PhoneService = st.selectbox("Phone Service", ["Yes","No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
        MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    with col2:
        OnlineSecurity = st.selectbox("Online Security", ["Yes","No","No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes","No","No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes","No","No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes","No","No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes","No"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"
        ])
        

    if st.button("Predict Churn"):

        payload = {
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }

        res = requests.post(f"{API}/predict", json=payload).json()

        prob = res["churn_probability"]
        risk = res["risk_level"]

        st.metric("Churn Probability", f"{prob:.2%}")
        st.success(f"Risk Level: {risk}")

        st.subheader("🧠 SHAP Explanation")

        #shap_res = requests.post(f"{API}/shap", json=payload).json()
        #img = base64.b64decode(shap_res["image"])
        #st.image(img)
        res = requests.post(f"{API}/shap", json=payload)

        print(res.status_code)
        print(res.text)

        data = res.json()

        if "error" in data:
            st.error(data["error"])
        else:
            img = base64.b64decode(data["image"])
            st.image(img)

# =========================================================
# PAGE 2 — BATCH UPLOAD
# =========================================================
elif page == "Batch Upload":

    st.header("📂 Batch Customer Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        files = {"file": file.getvalue()}
        res = requests.post(f"{API}/batch_predict", files=files)
        df = pd.DataFrame(res.json())

        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", csv, "churn_predictions.csv")

# =========================================================
# PAGE 3 — MODEL INSIGHTS
# =========================================================
else:

    st.header("📈 Model Insights")

    st.image("../models/roc_curve.png", caption="ROC Curve")
    st.image("../models/confusion_matrix.png", caption="Confusion Matrix")
    st.image("../models/feature_importance.png", caption="Feature Importance")

    st.markdown("""
    ### 🔍 Key Churn Drivers
    - Month-to-month contracts increase churn risk  
    - Fiber optic users show higher churn  
    - Lower tenure customers churn more  
    - Higher monthly charges increase churn  
    """)