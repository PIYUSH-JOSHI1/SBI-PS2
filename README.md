# SBI Life Retention Prediction System

This project implements an AI-driven customer retention prediction system for SBI Life Insurance, featuring personalized insights and user activity tracking.

## Features

- Customer retention prediction using machine learning
- Interactive Streamlit dashboard
- User activity tracking
- Real-time analytics visualization
- Synthetic data generation for testing

## Project Structure

```
sbi-life-retention-prediction/
├── src/
│   ├── data_generation.py    # Generates synthetic training data
│   ├── model_training.py     # ML model training and evaluation
│   ├── app.py               # Streamlit web application
│   └── user_activity_tracking.py  # User interaction logging
├── data/
│   ├── train_data.csv       # Training dataset
│   └── test_data.csv        # Test dataset
├── models/
│   └── policy_retention_model.joblib  # Trained ML model
├── logs/                    # User activity logs
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate synthetic data:
   ```bash
   python src/data_generation.py
   ```

3. Train the model:
   ```bash
   python src/model_training.py
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run src/app.py
   ```

## Usage

1. Navigate to the Prediction tab to input customer details and get retention predictions
2. View analytics and trends in the Analytics tab
3. Track user activity and interactions in the User Activity tab

## Features

- **Prediction System**
  - Customer retention probability calculation
  - Feature importance visualization
  - Risk assessment and recommendations

- **Analytics Dashboard**
  - Age distribution analysis
  - Income vs Premium insights
  - Occupation risk analysis
  - Payment frequency distribution

- **User Activity Tracking**
  - Session-based tracking
  - Interaction logging
  - Activity timeline visualization