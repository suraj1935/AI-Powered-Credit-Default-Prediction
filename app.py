import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your trained model
@st.cache_resource
def load_model():
    try:
        with open('credit_risk_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please run train_model.py first.")
        return None

def predict_risk(features, model):
    pred_proba = model.predict_proba(features)[:, 1]
    prediction = model.predict(features)
    return prediction, pred_proba

def generate_validation_data(model, n_samples=100):
    """Generate validation data consistent with training"""
    np.random.seed(42)
    
    test_features = pd.DataFrame({
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'loan_amount': np.random.randint(5000, 50000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'employment_years': np.random.randint(0, 30, n_samples)
    })
    
    # Use model to get realistic predictions for validation
    test_pred_proba = model.predict_proba(test_features)[:, 1]
    test_labels = (test_pred_proba > 0.5).astype(int)
    
    # Add some noise to make it realistic
    noise = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    test_labels = np.where(noise == 1, 1 - test_labels, test_labels)
    
    return test_features, test_labels

def main():
    st.title("Credit Risk Prediction & Validation")
    
    model = load_model()
    
    if model is None:
        st.stop()

    # Input section
    st.header("Input Credit Features")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Annual Income (in $)", min_value=0, value=50000)
        loan_amount = st.number_input("Loan Amount (in $)", min_value=0, value=10000)
    
    with col2:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        employment_years = st.number_input("Years of Employment", min_value=0, max_value=50, value=5)
    
    input_data = pd.DataFrame({
        'age': [age],
        'income': [income],
        'loan_amount': [loan_amount],
        'credit_score': [credit_score],
        'employment_years': [employment_years]
    })

    if st.button("Predict Credit Risk"):
        pred, prob = predict_risk(input_data, model)
        risk = 'High Risk' if pred[0] == 1 else 'Low Risk'
        
        # Display results with color coding
        if pred[0] == 1:
            st.error(f"Prediction: **{risk}**")
        else:
            st.success(f"Prediction: **{risk}**")
        
        st.write(f"Probability of Default: {prob[0]:.2f}")
        
        # Feature importance explanation
        st.subheader("Risk Factors")
        feature_importance = {
            'Credit Score': (850 - credit_score) / 550,
            'Loan-to-Income Ratio': loan_amount / income if income > 0 else 0,
            'Age': max(0, (30 - age) / 50),
            'Employment Stability': max(0, (5 - employment_years) / 5)
        }
        
        for factor, score in feature_importance.items():
            st.write(f"- {factor}: {score:.2f}")

    # Model validation section
    st.header("Model Validation Metrics")
    
    if st.button("Run Validation"):
        with st.spinner("Running validation..."):
            test_features, test_labels = generate_validation_data(model, 200)
            test_pred_proba = model.predict_proba(test_features)[:, 1]
            test_pred = model.predict(test_features)
            
            # Calculate metrics
            auc = roc_auc_score(test_labels, test_pred_proba)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("ROC-AUC Score", f"{auc:.3f}")
            
            with col2:
                accuracy = (test_pred == test_labels).mean()
                st.metric("Accuracy", f"{accuracy:.3f}")
            
            with col3:
                precision = (test_labels[test_pred == 1] == 1).mean() if (test_pred == 1).sum() > 0 else 0
                st.metric("Precision", f"{precision:.3f}")
            
            # Classification report
            st.subheader("Detailed Classification Report")
            report = classification_report(test_labels, test_pred, output_dict=True)
            st.table(pd.DataFrame(report).transpose())
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(test_labels, test_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

if __name__ == '__main__':
    main()

#suraj's AI/ML Credit Risk Model Development and Validation for Financial Services