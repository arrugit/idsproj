import streamlit as st
import pandas as pd
import pickle

# Load model, scaler, and feature names
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Streamlit app
st.title("Credit Card Fraud Detection App")

# Introduction
st.header("Introduction")
st.write("""
This app uses a Logistic Regression model to detect fraudulent credit card transactions.
The dataset includes transaction details like amount and time, with a focus on identifying rare fraud cases.
Explore the EDA insights and make predictions using the interactive tool below.
""")

# EDA Section
st.header("Exploratory Data Analysis")
st.write("Key insights from the dataset are visualized below.")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Fraud Distribution")
    st.image('fraud_count.png', caption='Count of Fraud vs Non-Fraud')
    st.subheader("Amount Distribution")
    st.image('amount_hist.png', caption='Transaction Amount Distribution')
with col2:
    st.subheader("Fraud Percentage")
    st.image('fraud_pie.png', caption='Percentage of Fraud Transactions')
    st.subheader("Amount by Fraud Status")
    st.image('amount_boxplot.png', caption='Box Plot of Amount by Fraud')

st.subheader("Correlation Analysis")
st.image('correlation_heatmap.png', caption='Correlation Heatmap of Numerical Features')
st.subheader("Advanced Visualizations")
st.image('amount_kde.png', caption='KDE Plot of Amount by Fraud Status')
st.image('top_feature_violin.png', caption='Violin Plot of Top Feature by Fraud Status')

# Model Section
st.header("Fraud Prediction")
st.write("""
The Logistic Regression model is trained with balanced class weights to handle the imbalanced dataset.
It achieves high accuracy and a reasonable F1-score for fraud detection.
Enter transaction details below to predict fraud.
""")

# User input
st.subheader("Enter Transaction Details")
input_data = {}
input_data['amt'] = st.number_input("Transaction Amount ($)", min_value=0.0, value=0.0)
input_data['hour'] = st.slider("Transaction Hour (0-23)", min_value=0, max_value=23, value=12)
category = st.selectbox("Merchant Category", ['grocery_pos', 'shopping_net', 'misc_net', 'gas_transport', 'others'])

# Prepare input
if st.button("Predict Fraud"):
    input_df = pd.DataFrame({col: [0] for col in feature_names})
    input_df['amt'] = input_data['amt']
    input_df['hour'] = input_data['hour']
    if category != 'others':
        cat_col = f"category_{category}"
        if cat_col in input_df.columns:
            input_df[cat_col] = 1
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    
    # Display
    result = "Fraudulent" if prediction == 1 else "Non-Fraudulent"
    st.write(f"**Prediction**: {result}")
    st.write(f"**Fraud Probability**: {prob:.2%}")

# Conclusion
st.header("Conclusion")
st.write("""
Key findings:
- Fraud transactions are extremely rare (~0.5% of the dataset).
- Transaction amount and hour are significant predictors.
- The Logistic Regression model provides reliable predictions despite class imbalance.
- Future work could explore ensemble methods or oversampling to improve fraud detection.
""")