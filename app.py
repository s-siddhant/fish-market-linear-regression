from flask import Flask, request, render_template
import joblib

# Load the saved model
model = joblib.load("fish_market_model.pkl")

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # Get user inputs from the form
    species = float(request.form["species"])
    length1 = float(request.form["length1"])
    length2 = float(request.form["length2"])
    length3 = float(request.form["length3"])
    height = float(request.form["height"])
    width = float(request.form["width"])

    # Make prediction
    features = [[species, length1, length2, length3, height, width]]
    predicted_weight = model.predict(features)[0]

    # Return the prediction to the frontend
    return f"Predicted Weight: {predicted_weight:.2f} grams"

# Run the app
if __name__ == "__main__":
    app.run(debug=True)