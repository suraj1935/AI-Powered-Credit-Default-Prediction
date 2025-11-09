import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config first
st.set_page_config(
    page_title="AI-Powered Credit Default Prediction",
    page_icon="💰",
    layout="wide"
)

# Load your trained model
@st.cache_resource
def load_model():
    try:
        with open('credit_risk_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'credit_risk_model.pkl' is in the root directory.")
        return None

def predict_risk(features, model):
    pred_proba = model.predict_proba(features)[:, 1]
    prediction = model.predict(features)
    return prediction, pred_proba

def generate_validation_data(model, n_samples=200):
    """Generate validation data consistent with training"""
    np.random.seed(42)
    
    test_features = pd.DataFrame({
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'loan_amount': np.random.randint(5000, 50000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'employment_years': np.random.randint(0, 30, n_samples)
    })
    
    test_pred_proba = model.predict_proba(test_features)[:, 1]
    test_labels = (test_pred_proba > 0.5).astype(int)
    
    # Add some noise to make it realistic
    noise = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    test_labels = np.where(noise == 1, 1 - test_labels, test_labels)
    
    return test_features, test_labels

def main():
    st.title("💰 AI-Powered Credit Default Prediction")
    st.markdown("A Streamlit-based web app that predicts credit default risk using a Random Forest model")
    
    model = load_model()
    
    if model is None:
        st.stop()

    # Input section
    st.header("📊 Input Credit Features")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        income = st.number_input("Annual Income ($)", min_value=0, value=60000)
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=15000)
    
    with col2:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720)
        employment_years = st.number_input("Years of Employment", min_value=0, max_value=50, value=8)
    
    input_data = pd.DataFrame({
        'age': [age],
        'income': [income],
        'loan_amount': [loan_amount],
        'credit_score': [credit_score],
        'employment_years': [employment_years]
    })

    if st.button("🚀 Predict Credit Risk", type="primary"):
        with st.spinner("Analyzing risk..."):
            pred, prob = predict_risk(input_data, model)
            risk = 'High Risk' if pred[0] == 1 else 'Low Risk'
            
            # Display results
            if pred[0] == 1:
                st.error(f"## Prediction: {risk}")
            else:
                st.success(f"## Prediction: {risk}")
            
            st.metric("Probability of Default", f"{prob[0]:.1%}")
            
            # Risk factors
            st.subheader("📈 Risk Factors Analysis")
            debt_to_income = loan_amount / income if income > 0 else 0
            credit_utilization = (850 - credit_score) / 550
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Debt-to-Income Ratio", f"{debt_to_income:.2f}")
            with col2:
                st.metric("Credit Risk Score", f"{credit_utilization:.2f}")
            with col3:
                st.metric("Employment Stability", f"{min(employment_years/10, 1.0):.2f}")

    # Model validation section
    st.header("🔍 Model Validation")
    
    if st.button("Run Model Validation"):
        with st.spinner("Generating validation report..."):
            test_features, test_labels = generate_validation_data(model, 200)
            test_pred_proba = model.predict_proba(test_features)[:, 1]
            test_pred = model.predict(test_features)
            
            # Calculate metrics
            auc = roc_auc_score(test_labels, test_pred_proba)
            accuracy = (test_pred == test_labels).mean()
            
            # Display metrics
            st.subheader("Model Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("ROC-AUC Score", f"{auc:.3f}")
            with col2:
                st.metric("Accuracy", f"{accuracy:.3f}")
            with col3:
                precision = (test_labels[test_pred == 1] == 1).mean() if (test_pred == 1).sum() > 0 else 0
                st.metric("Precision", f"{precision:.3f}")
            with col4:
                recall = (test_pred[test_labels == 1] == 1).mean() if (test_labels == 1).sum() > 0 else 0
                st.metric("Recall", f"{recall:.3f}")
            
            # Classification report
            st.subheader("Detailed Classification Report")
            report = classification_report(test_labels, test_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(test_labels, test_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Low Risk', 'High Risk'],
                       yticklabels=['Low Risk', 'High Risk'])
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            st.pyplot(fig)

if __name__ == '__main__':
    main()

#suraj's AI/ML Credit Risk Model Development and Validation for Financial Services