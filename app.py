import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('housing_sentiment_model.pkl')

# Title of the app
st.title('Housing Price Prediction with Sentiment Analysis')

# Input fields for user input
population = st.number_input("Population", min_value=0.0)
number_beds = st.number_input("Number of Bedrooms", min_value=1)
number_baths = st.number_input("Number of Bathrooms", min_value=1)
latitude = st.number_input("Latitude")
longitude = st.number_input("Longitude")
median_income = st.number_input("Median Family Income", min_value=0.0)
sentiment_score = st.number_input("Sentiment Score", min_value=-1.0, max_value=1.0)
weighted_sentiment = st.number_input("Weighted Sentiment Score", min_value=-1.0, max_value=1.0)
neutral_sentiment = st.checkbox("Neutral Sentiment Category")
positive_sentiment = st.checkbox("Positive Sentiment Category")

# Prepare the input data in the correct format for prediction
input_data = pd.DataFrame({
    'Population': [population],
    'Number_Beds': [number_beds],
    'Number_Baths': [number_baths],
    'Latitude': [latitude],
    'Longitude': [longitude],
    'Median_Family_Income': [median_income],
    'sentiment_score': [sentiment_score],
    'Weighted_Sentiment_Score': [weighted_sentiment],
    'sentiment_category_Neutral': [int(neutral_sentiment)],  # Convert True/False to 1/0
    'sentiment_category_Positive': [int(positive_sentiment)]  # Convert True/False to 1/0
})

# Prediction button
if st.button("Predict Housing Price"):
    # Make the prediction
    prediction = model.predict(input_data)
    
    # Show the result
    st.subheader(f"Predicted Price: ${prediction[0]:,.2f}")

