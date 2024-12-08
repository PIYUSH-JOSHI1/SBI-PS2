import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate synthetic customer data
    data = {
        'age': np.random.normal(40, 10, n_samples).astype(int),
        'income': np.random.normal(60000, 20000, n_samples),
        'policy_term': np.random.choice([5, 10, 15, 20, 25], n_samples),
        'premium_amount': np.random.normal(5000, 2000, n_samples),
        'payment_frequency': np.random.choice(['Monthly', 'Quarterly', 'Yearly'], n_samples),
        'dependents': np.random.randint(0, 5, n_samples),
        'occupation_risk': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'previous_claims': np.random.randint(0, 3, n_samples),
        'credit_score': np.random.normal(700, 50, n_samples)
    }
    
    # Calculate retention probability based on features
    retention_prob = (
        0.7 +
        0.1 * (data['income'] > 70000) +
        0.1 * (data['age'] > 35) +
        0.05 * (data['previous_claims'] < 2) +
        0.05 * (data['credit_score'] > 700)
    )
    
    # Generate retention labels
    data['retention'] = (np.random.random(n_samples) < retention_prob).astype(int)
    
    df = pd.DataFrame(data)
    
    # Save full dataset
    df.to_csv('data/full_dataset.csv', index=False)
    
    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save train and test datasets
    train_df.to_csv('data/train_data.csv', index=False)
    test_df.to_csv('data/test_data.csv', index=False)
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = generate_synthetic_data()