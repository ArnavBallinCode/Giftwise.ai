import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load and prepare the final dataset
data = pd.read_csv('gift_dataset_final.csv')
X = data.iloc[:, :-1]  # All columns except the last one
y = data['ideal_gift']  # Last column is the target

# Encode the personality traits
encoder = LabelEncoder()
X_encoded = X.apply(encoder.fit_transform)

# Train the model
model = LogisticRegression(max_iter=200)  # Increased max_iter to avoid convergence warning
model.fit(X_encoded, y)

# Save the trained model
joblib.dump(model, 'gift_recommendation_model.pkl')

# Predefined gift items
gift_items = [
    "Wireless Headphones", "Smart Watch", "Kindle", "Gaming Chair", "Art Supplies",
    "Cooking Kit", "Fitness Tracker", "Luxury Pen", "Customized Mug", "Board Game",
    "Portable Speaker", "Photo Album", "Plant Set", "Puzzle", "Backpack",
    "Subscription Box", "Desk Organizer", "E-Book Subscription", "Travel Kit", "Coffee Maker"
]

# Function to rank gifts based on survey responses
def rank_gifts(responses):
    # Convert responses to a format suitable for the model
    input_data = pd.DataFrame([responses], columns=[f'question{i}' for i in range(1, 26)])  # Adjusted to 4 questions
    try:
        input_encoded = input_data.apply(encoder.transform)
        predictions = model.predict(input_encoded)
    except ValueError as e:
        # Handle unseen labels by returning a default recommendation
        st.error(f"Error: {e}. Returning default recommendations.")
        return ["Gift Card", "Surprise Box", "Mystery Gift"]
    
    # Rank gifts based on model predictions
    ranked_gifts = [gift_items[i] for i in predictions.argsort()[:5]]
    return ranked_gifts

# Streamlit app setup
st.title("Personalized Gift Recommendation AI")
st.write("Find the perfect gift based on your personality and preferences. Simply take our survey, and we'll provide tailored gift suggestions just for you!")

preferences = []
preferences.append(st.selectbox('Environment', ['Indoor', 'Outdoor']))
preferences.append(st.selectbox('Likes Music', ['Yes', 'No']))
preferences.append(st.selectbox('Music Genre', ['Pop', 'Rock']))
preferences.append(st.selectbox('Likes Reading', ['Yes', 'No']))

if st.button('Get Recommendation'):
    recommendation = rank_gifts(preferences)
    st.write(f"We recommend: {recommendation}")
