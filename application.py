from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd  # ✅ ensure pandas is included in requirements.txt

# Initialize Flask app
application = Flask(__name__)

# Load KMeans model
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")  # ✅ load the same scaler used in training

# Health check endpoint
@application.route('/')
def home():
    return "✅ LG Customer Segmentation API is live!"

# Customer segmentation endpoint
@application.route('/segment', methods=['POST'])
def segment():
    try:
        data = request.get_json()

        # Required features (must match training set)
        required_features = [
            "Age","Income","LoyaltyScore","OnlineEngagement",
            "DaysSinceLastPurchase","QuantityPurchased",
            "PreferenceScore","WillingnessToPay"
        ]

        # Check if all required features exist in request
        if not all(feature in data for feature in required_features):
            missing = [f for f in required_features if f not in data]
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Convert input to DataFrame for KMeans
        features = pd.DataFrame([{
            "Age": data["Age"],
            "Income": data["Income"],
            "LoyaltyScore": data["LoyaltyScore"],
            "OnlineEngagement": data["OnlineEngagement"],
            "DaysSinceLastPurchase": data["DaysSinceLastPurchase"],
            "QuantityPurchased": data["QuantityPurchased"],
            "PreferenceScore": data["PreferenceScore"],
            "WillingnessToPay": data["WillingnessToPay"]
        }])

        # Scale input features just like training
        features_scaled = scaler.transform(features)

        # Predict cluster
        cluster = kmeans.predict(features_scaled)[0]
        return jsonify({"CustomerSegment": int(cluster)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run locally
if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000)
