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

        #st.metric("Churn Probability", f"{prob:.2%}")

        st.subheader("Churn Probability")

        st.progress(prob)

        st.write(f"### {prob*100:.2f}%")

        if risk=="Low":
            st.success("Low Risk Customer")

        elif risk=="Medium":
            st.warning("Medium Risk Customer")

        else:
            st.error("High Risk Customer")
        #st.success(f"Risk Level: {risk}")

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
            st.markdown(
            """
            The SHAP waterfall plot shows how each feature contributes to the predicted churn probability.

            🔴 Red features increase churn risk  
            🔵 Blue features reduce churn risk
            """
            )

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

    import os

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    roc_path = os.path.join(BASE_DIR, "models", "ROC_curve.png")
    CM_path  = os.path.join(BASE_DIR, "models", "confusion_matrix.png")
    FI_path  = os.path.join(BASE_DIR, "models", "feature_importance.png")
    SHAP_path = os.path.join(BASE_DIR, "models", "shap_summary.png")
    
    st.set_page_config(layout="wide")

    st.header("📶 Model Insights Dashboard")
    #st.markdown("Performance analysis and explainability of the churn prediction model")

    st.divider()

    # ===============================
    # 1️⃣ MODEL PERFORMANCE METRICS
    # ===============================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("ROC AUC", "0.843")
    col2.metric("Accuracy", "76.4%")
    col3.metric("Recall", "74.9%")
    col4.metric("F1 Score", "0.63")

    st.divider()

    # ===============================
    # 2️⃣ MODEL PERFORMANCE VISUALS
    # ===============================

    st.subheader("📈 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ROC Curve")
        st.image(roc_path, use_container_width=True)

    with col2:
        st.markdown("#### Confusion Matrix")
        st.image(CM_path, use_container_width=True)

    st.divider()

    # ===============================
    # 3️⃣ FEATURE IMPORTANCE
    # ===============================

    st.subheader("🔍 Feature Intelligence")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Feature Importance")
        st.image(FI_path, use_container_width=True)

    with col2:
        st.markdown("#### SHAP Summary Plot")
        st.image(SHAP_path, use_container_width=True)

    st.divider()

    # ===============================
    # 5️⃣ BUSINESS INSIGHTS
    # ===============================

    st.subheader("💡 Business Insights")

    st.info(
    """
    ### 🔺 Key Drivers of Churn
    • Month-to-month contracts  
    • High monthly charges  
    • Fiber optic internet users  

    ### 🔻 Drivers of Retention
    • Long-term contracts (1–2 years)  
    • Customers with tech support  
    • Long tenure customers  

    ### 📈 Business Impact
    The model can identify high-risk customers early, allowing telecom companies to deploy retention strategies before churn occurs.
    """
    )

    st.divider()

    # ===============================
    # 6️⃣ RETENTION RECOMMENDATION ENGINE
    # ===============================

    st.subheader("🎯 Retention Strategy Recommender")

    risk_level = st.selectbox(
        "Select Customer Risk Level",
        ["Low Risk", "Medium Risk", "High Risk"]
    )

    if risk_level == "Low Risk":

        st.success("""
        **Suggested Strategy:** \n
        • Offer loyalty rewards  
        • Promote bundled services  
        • Upsell long-term contracts
        """)

    elif risk_level == "Medium Risk":

        st.warning("""
        **Suggested Strategy:** \n
        • Offer contract upgrade discounts  
        • Provide service bundles  
        • Improve customer support engagement
        """)

    else:

        st.error("""
        **Suggested Strategy:** \n
        • Offer targeted retention discounts  
        • Provide priority technical support  
        • Assign customer success manager
        """)
        