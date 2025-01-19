import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

class GiftRecommendationModel:
    def __init__(self, dataset_path='gift_dataset_final.csv'):  # Set default dataset path
        self.dataset_path = dataset_path
        self.model = LogisticRegression(max_iter=500)  # Increased max_iter
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()  # Added scaler
        self.train_model()

    def train_model(self):
        # Load and prepare the dataset
        data = pd.read_csv(self.dataset_path)
        X = data.iloc[:, :-1]  # All columns except the last one
        y = data['ideal_gift']  # Last column is the target

        # Encode the personality traits
        X_encoded = X.apply(self.encoder.fit_transform)

        # Scale the features
        X_scaled = self.scaler.fit_transform(X_encoded)

        # Train the model
        self.model.fit(X_scaled, y)

        # Save the trained model
        joblib.dump(self.model, 'gift_recommendation_model.pkl')

    def predict(self, responses):
        # Convert responses to a format suitable for the model
        input_data = pd.DataFrame([responses], columns=self.encoder.classes_)
        input_encoded = input_data.apply(self.encoder.transform)
        input_scaled = self.scaler.transform(input_encoded)  # Scale the input
        predictions = self.model.predict(input_scaled)
        return predictions

    def rank_gifts(self, responses):
        try:
            predictions = self.predict(responses)
            ranked_gifts = [self.gift_items[i] for i in predictions.argsort()[:5]]
            return ranked_gifts
        except ValueError as e:
            # Handle unseen labels by returning a default recommendation
            print(f"Error: {e}. Returning default recommendations.")
            return ["Luxury Pen", "Smart Watch", "Wireless Headphones", "Cooking Kit", "Board Game"]
