import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pickle

def generate_sample_data(n_samples=1000):
    """Generate synthetic credit risk data for demonstration"""
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'loan_amount': np.random.randint(5000, 50000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'employment_years': np.random.randint(0, 30, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create synthetic target variable (credit risk)
    # Higher risk for: low credit score, high loan/income ratio, young age
    risk_score = (
        (850 - df['credit_score']) / 550 * 0.4 +
        (df['loan_amount'] / df['income']) * 0.3 +
        ((30 - df['age']) / 50 * 0.2).clip(0, 1) +
        ((5 - df['employment_years']) / 5 * 0.1).clip(0, 1)
    )
    
    df['default'] = (risk_score > 0.5).astype(int)
    
    return df

def train_credit_risk_model():
    # Generate sample data
    df = generate_sample_data(2000)
    
    # Features and target
    X = df[['age', 'income', 'loan_amount', 'credit_score', 'employment_years']]
    y = df['default']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("Model Performance:")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    with open('credit_risk_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("\nModel saved as 'credit_risk_model.pkl'")
    
    return model, X_test, y_test

if __name__ == '__main__':
    train_credit_risk_model()