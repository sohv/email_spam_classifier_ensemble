import streamlit as st
import numpy as np
from joblib import load
import pandas as pd

# page configuration
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide"
)

# load the model components
@st.cache_resource
def load_model():
    try:
        components = load('ensemble_model_components.joblib')
        return components
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_prediction_probability(predictions):
    spam_votes = sum(1 for pred in predictions if pred == 1)
    return spam_votes / len(predictions)

def predict_spam(subject, message, components):
    vectorizer = components['vectorizer']
    sgd_model = components['sgd_model']
    svm_model = components['svm_model']
    rf_model = components['rf_model']

    text = f"{subject} {message}"
    features = vectorizer.transform([text])
    
    pred_sgd = sgd_model.predict(features)[0]
    pred_svm = svm_model.predict(features)[0]
    pred_rf = rf_model.predict(features)[0]
    
    predictions = [pred_sgd, pred_svm, pred_rf]
    
    ensemble_pred = np.bincount(predictions).argmax()
    
    probability = get_prediction_probability(predictions)
    individual_results = {
        'Stochastic Gradient Descent': 'Spam' if pred_sgd == 1 else 'Ham',
        'Support Vector Machine': 'Spam' if pred_svm == 1 else 'Ham',
        'Random Forest': 'Spam' if pred_rf == 1 else 'Ham'
    }
    
    return ensemble_pred, probability, individual_results

def main():
    st.title("📧 Email Spam Detector")
    st.markdown("""
    This app uses an ensemble machine learning model to detect spam emails.
    Enter the email subject and message to check if it's spam or ham (legitimate).
    """)
    
    components = load_model()
    
    if components is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        subject = st.text_input("Email Subject:", key="subject")
        
    with col2:
        message = st.text_area("Email Message:", key="message", height=100)
    
    if st.button("Predict", type="primary"):
        if not subject and not message:
            st.warning("Please enter both subject and message.")
            return
        
        prediction, probability, individual_results = predict_spam(subject, message, components)
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Ensemble Prediction")
            if prediction == 1:
                st.error("📨 SPAM")
            else:
                st.success("✉️ HAM (Legitimate)")
        
        with col2:
            st.markdown("### Confidence Score")
            confidence = probability if prediction == 1 else (1 - probability)
            st.progress(confidence)
            st.write(f"{confidence:.2%} confident")
        
        with col3:
            st.markdown("### Individual Model Predictions")
            for model, result in individual_results.items():
                st.write(f"{model}: {result}")

        st.markdown("---")
        st.markdown("### Prediction Details")
        st.write("""
        - A confidence score close to 100% indicates strong agreement among the models
        - The ensemble prediction is based on majority voting from three models
        - Individual model predictions show how each model classified the email
        """)

if __name__ == "__main__":
    main()