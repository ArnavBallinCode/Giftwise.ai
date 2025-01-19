import streamlit as st
import pandas as pd
from model import GiftRecommendationModel  # Adjust the import based on your model

# Load the model
model = GiftRecommendationModel()

# Title
st.title("Personalized Gift Recommendation AI")

# Survey Form
st.header("Take the Survey")
responses = []
responses.append(st.selectbox('Environment', ['Indoor', 'Outdoor']))
responses.append(st.selectbox('Likes Music', ['Yes', 'No']))
responses.append(st.selectbox('Music Genre', ['Pop', 'Rock', 'Jazz']))
responses.append(st.selectbox('Likes Reading', ['Yes', 'No']))

if st.button("Get Recommendation"):
    recommendations = model.rank_gifts(responses)
    st.write("Your Recommendations:")
    for gift in recommendations:
        st.write(gift)
