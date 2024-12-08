import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def preprocess_data(df):
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # Label encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['payment_frequency', 'occupation_risk']
    
    for col in categorical_cols:
        df_processed[col] = le.fit_transform(df_processed[col])
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['age', 'income', 'premium_amount', 'credit_score']
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    return df_processed, scaler, le

def train_model():
    # Load training data
    train_data = pd.read_csv('../data/train_data.csv')
    
    # Preprocess data
    X_train_processed, scaler, le = preprocess_data(train_data.drop('retention', axis=1))
    y_train = train_data['retention']
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_processed, y_train)
    
    # Save model and preprocessors
    joblib.dump(model, '../models/policy_retention_model.joblib')
    joblib.dump(scaler, '../models/scaler.joblib')
    joblib.dump(le, '../models/label_encoder.joblib')
    
    # Evaluate on test data
    test_data = pd.read_csv('../data/test_data.csv')
    X_test_processed, _, _ = preprocess_data(test_data.drop('retention', axis=1))
    y_test = test_data['retention']
    
    # Generate and print classification report
    y_pred = model.predict(X_test_processed)
    print("\nModel Performance Report:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler, le

if __name__ == "__main__":
    train_model()