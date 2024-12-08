import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model_training import preprocess_data
import joblib
from user_activity_tracking import UserActivityTracker
import uuid
from datetime import datetime

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# Initialize activity tracker
tracker = UserActivityTracker()

# Load the trained model and preprocessors
@st.cache_resource
def load_model():
    model = joblib.load('../models/policy_retention_model.joblib')
    scaler = joblib.load('../models/scaler.joblib')
    le = joblib.load('../models/label_encoder.joblib')
    return model, scaler, le

def main():
    st.set_page_config(page_title="SBI Life Retention Prediction", layout="wide")
    
    st.title("SBI Life Insurance Retention Prediction System")
    
    # Track page visit
    tracker.log_activity(
        st.session_state.user_id,
        "page_visit",
        {"page": "main"}
    )
    
    tabs = st.tabs(["Prediction", "Analytics", "User Activity"])
    
    with tabs[0]:
        show_prediction_tab()
    
    with tabs[1]:
        show_analytics_tab()
    
    with tabs[2]:
        show_activity_tab()

def show_prediction_tab():
    st.header("Customer Retention Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=35)
        income = st.number_input("Annual Income", min_value=0, max_value=1000000, value=60000)
        policy_term = st.selectbox("Policy Term (Years)", [5, 10, 15, 20, 25])
        premium_amount = st.number_input("Premium Amount", min_value=1000, max_value=50000, value=5000)
        payment_frequency = st.selectbox("Payment Frequency", ["Monthly", "Quarterly", "Yearly"])
    
    with col2:
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
        occupation_risk = st.selectbox("Occupation Risk", ["Low", "Medium", "High"])
        previous_claims = st.number_input("Previous Claims", min_value=0, max_value=5, value=0)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    
    if st.button("Predict Retention"):
        # Track prediction attempt
        tracker.log_activity(
            st.session_state.user_id,
            "prediction_attempt",
            {
                "age": age,
                "income": income,
                "policy_term": policy_term
            }
        )
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age],
            'income': [income],
            'policy_term': [policy_term],
            'premium_amount': [premium_amount],
            'payment_frequency': [payment_frequency],
            'dependents': [dependents],
            'occupation_risk': [occupation_risk],
            'previous_claims': [previous_claims],
            'credit_score': [credit_score]
        })
        
        # Load model and make prediction
        model, scaler, le = load_model()
        processed_input, _, _ = preprocess_data(input_data)
        prediction = model.predict_proba(processed_input)[0]
        
        # Display prediction
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            retention_prob = prediction[1] * 100
            st.metric("Retention Probability", f"{retention_prob:.1f}%")
            
            if retention_prob >= 70:
                st.success("High retention probability! Consider offering premium services.")
            elif retention_prob >= 40:
                st.warning("Moderate retention risk. Consider personalized retention strategies.")
            else:
                st.error("High retention risk! Immediate attention required.")
        
        with col2:
            # Feature importance plot
            feature_importance = pd.DataFrame({
                'feature': input_data.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title='Feature Importance'
            )
            st.plotly_chart(fig)

def show_analytics_tab():
    st.header("Customer Analytics Dashboard")
    
    # Load and display some analytics from the training data
    try:
        train_data = pd.read_csv('../data/train_data.csv')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig_age = px.histogram(
                train_data,
                x='age',
                color='retention',
                title='Age Distribution by Retention'
            )
            st.plotly_chart(fig_age)
            
            # Income vs Premium
            fig_income = px.scatter(
                train_data,
                x='income',
                y='premium_amount',
                color='retention',
                title='Income vs Premium Amount'
            )
            st.plotly_chart(fig_income)
        
        with col2:
            # Retention by occupation risk
            retention_by_risk = train_data.groupby('occupation_risk')['retention'].mean()
            fig_risk = px.bar(
                retention_by_risk,
                title='Retention Rate by Occupation Risk'
            )
            st.plotly_chart(fig_risk)
            
            # Payment frequency distribution
            fig_payment = px.pie(
                train_data,
                names='payment_frequency',
                title='Payment Frequency Distribution'
            )
            st.plotly_chart(fig_payment)
            
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")

def show_activity_tab():
    st.header("User Activity Log")
    
    # Display recent activities for the current user
    activities = tracker.get_user_activities(st.session_state.user_id)
    
    if activities:
        activity_df = pd.DataFrame(activities)
        activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp'])
        activity_df = activity_df.sort_values('timestamp', ascending=False)
        
        st.dataframe(
            activity_df[['timestamp', 'action', 'details']],
            use_container_width=True
        )
    else:
        st.info("No activity recorded yet.")

if __name__ == "__main__":
    main()