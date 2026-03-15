#    Customer Churn Prediction

A comprehensive machine learning solution to predict customer churn in telecom/service industries. This project combines advanced ML models, feature engineering, and explainable AI to help businesses identify at-risk customers and reduce revenue loss.

**Live Demo & Deployment:**
- 🚀 **Frontend:** [https://customer-churn-prediction-front.streamlit.app/](https://customer-churn-prediction-front.streamlit.app/) (Streamlit Cloud)
- 🔧 **Backend API:** [https://customer-churn-prediction-ct21.onrender.com/](https://customer-churn-prediction-ct21.onrender.com/) (Render)

---

## 🎯 Features

- **Individual Customer Prediction**: Predict churn probability for single customers with detailed risk assessment
- **Batch Prediction**: Upload CSV files to predict churn for multiple customers simultaneously
- **Risk Categorization**: Automatic risk level classification (Low, Medium, High) based on probability thresholds
- **SHAP Explainability**: Waterfall plots showing which features influence each prediction
- **Model Comparison**: Analysis of multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- **Interactive Dashboard**: User-friendly Streamlit interface for data exploration and predictions
- **RESTful API**: FastAPI backend with CORS support for easy integration

---

## 🏗️ Project Structure

```
Customer-Churn-Prediction/
├── backend/                    # FastAPI Backend
│   ├── main.py                # Main API application
│   ├── schemas.py             # Pydantic schemas for data validation
│   ├── feature_engineering.py # Feature transformation pipeline
│   ├── utils.py               # Prediction & SHAP utilities
│   └── __init__.py
├── frontend/                   # Streamlit Frontend
│   └── app.py                 # Main dashboard application
├── data/                       # Dataset directory
│   ├── Telco-Customer-Churn-Raw.csv      # Raw dataset
│   ├── Telco_Churn_Cleaned.csv           # Cleaned dataset
│   ├── X_cleaned.csv                     # Features
│   └── y_cleaned.csv                     # Target variable
├── models/                     # Trained ML models
├── notebooks/                  # Jupyter notebooks
│   ├── 01_eda.ipynb                      # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb      # Feature engineering
│   ├── 03_logistic_regression.ipynb      # Logistic Regression model
│   ├── 04_decision_tree.ipynb            # Decision Tree model
│   ├── 05_random_forest.ipynb            # Random Forest model
│   ├── 06_xgboost.ipynb                  # XGBoost model
│   ├── 07_model_comparison.ipynb         # Model evaluation & comparison
│   └── 08_model_explainability_shap.ipynb # SHAP analysis
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 💻 Tech Stack

### Backend
- **Framework**: FastAPI (modern async web framework)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Deployment**: Render

### Frontend
- **Framework**: Streamlit (interactive data app)
- **Data Visualization**: Plotly, Matplotlib, Seaborn
- **HTTP Client**: Requests
- **Deployment**: Streamlit Cloud

### Data Science & ML
- **Models**: Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Feature Engineering**: Custom transformations in backend
- **Evaluation**: Scikit-learn metrics

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Git
- pip or conda

### Local Development Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

#### 2. Create Virtual Environment
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run Backend Server
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
Backend will be available at: `http://localhost:8000`

#### 5. Run Frontend (in new terminal)
```bash
streamlit run frontend/app.py
```
Frontend will be available at: `http://localhost:8501`

---

## 📖 Usage Guide

### Web Dashboard (Streamlit)

The dashboard provides three main sections:

#### 1. **Customer Prediction**
- Fill in customer attributes
- Click "Predict Churn" to see:
  - Churn probability (0-1)
  - Risk level categorization
  - SHAP explainability plot

#### 2. **Batch Upload**
- Upload a CSV file with multiple customer records
- Returns predictions and risk levels for all customers
- Download results as CSV

#### 3. **Model Insights**
- Explore model performance metrics
- View feature importance and model explainability
- Understand business insights and retention strategy

### API Endpoints

#### Health Check
```bash
GET /
```
Response:
```json
{"message": "Churn Prediction API Running"}
```

#### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 24,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "Yes",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "One year",
  "PaperlessBilling": "No",
  "PaymentMethod": "Bank transfer (automatic)",
  "MonthlyCharges": 65.5,
  "TotalCharges": 1570.0
}
```

Response:
```json
{
  "churn_probability": 0.25,
  "risk_level": "Low"
}
```

#### Batch Prediction
```bash
POST /batch_predict
Content-Type: multipart/form-data

file: <CSV file with customer data>
```

Response:
```json
[
  {
    "gender": "Male",
    "tenure": 24,
    "churn_probability": 0.25,
    "risk_level": "Low"
  }
]
```

#### SHAP Explainability
```bash
POST /shap
Content-Type: application/json

{
  "gender": "Male",
  "SeniorCitizen": 0,
  ...
}
```

Returns: Base64-encoded SHAP waterfall plot image

---

## 🎓 Data Overview

### Dataset: Telco Customer Churn
- **Samples**: ~7,000+ customer records
- **Features**: 19 customer attributes
- **Target**: Binary (Churn: Yes/No)
- **Features Include**:
  - Demographics: Gender, Age, Partner status
  - Services: Internet type, Phone service, Online security
  - Account Info: Tenure, Contract type, Payment method
  - Charges: Monthly and total charges

---

## 📊 Model Performance

The project evaluates multiple ML models:

|       Model      | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
Logistic Regression	  0.735	      0.501	     0.791	  0.613	     0.841
Random Forest	      0.764	      0.539	     0.749	  0.627	     0.843	
Decision Tree	      0.754	      0.526	     0.751	  0.619	     0.837	
XGBoost	              0.804	      0.674	     0.508	  0.579	     0.848

*See `notebooks/07_model_comparison.ipynb` for detailed metrics*

---

## 🔍 Model Explainability (SHAP)

The project uses SHAP (SHapley Additive exPlanations) to provide transparent predictions:

- **Waterfall Plots**: Show how each feature contributes to pushing the prediction away from the base value
- **Feature Impact**: Identifies which customer attributes are most influential in predicting churn
- **Model Transparency**: Helps stakeholders understand "why" the model made a specific prediction

See `notebooks/08_model_explainability_shap.ipynb` for detailed analysis.

---

## 🌐 Deployment

### Backend Deployment (Render)
1. Connect GitHub repository to Render
2. Create new Web Service with:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`
3. Add environment variables as needed
4. Deploy

### Frontend Deployment (Streamlit Cloud)
1. Push code to GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub account and select repository
4. Configure app location as `frontend/app.py`
5. Deploy

---

## 📁 Dataset Features

**Customer Demographics**
- Gender
- Age (SeniorCitizen status)
- Partner status
- Dependents

**Services Subscribed**
- Phone Service
- Internet Service (DSL, Fiber optic, None)
- Online Security, Online Backup
- Device Protection, Tech Support
- Streaming TV, Streaming Movies

**Account Information**
- Tenure (months)
- Contract type (Month-to-month, One year, Two year)
- Billing preference (Paperless)
- Payment method
- Monthly and Total Charges

---

## 🛠️ Development & Contributing

### Setup for Contributors
```bash
# Clone repo
git clone https://github.com/your-username/Customer-Churn-Prediction.git

# Create feature branch
git checkout -b feature/your-feature

# Make changes and test
python -m pytest  # if tests exist

# Commit and push
git add .
git commit -m "Add feature description"
git push origin feature/your-feature

# Create Pull Request
```

### Code Structure
- Backend logic in `backend/` for API endpoints and ML utilities
- Frontend UI in `frontend/` for Streamlit dashboard
- Notebooks in `notebooks/` for exploratory analysis and model training
- Data in `data/` directory

---

## 🎯 Key Insights

- Customers with month-to-month contracts have significantly higher churn rates
- Short tenure is a strong predictor of churn
- Internet fiber optic users show higher churn compared to DSL
- Having tech support and security services reduces churn probability
- Total charges and contract duration are key protective factors

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👥 Contact & Support

For questions, suggestions, or issues:
- Open an Issue on GitHub
- Submit a Pull Request
- Check existing documentation in the notebooks

---

## 🎉 Acknowledgments

- Dataset: IBM Telco Customer Churn
- Libraries: FastAPI, Streamlit, Scikit-learn, SHAP, XGBoost
- Deployment: Render, Streamlit Cloud

---

**Last Updated**: March 2026
**Status**: ✅ Active & Maintained
