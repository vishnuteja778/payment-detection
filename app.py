from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model
# We define the path relative to the current script
model_path = 'model.pkl'

# Initialize model variable
model = None

def load_prediction_model():
    global model
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Warning: model.pkl not found. Please train the model first.")

load_prediction_model()

@app.route('/')
def home():
    """Landing page with platform overview and 3D animations"""
    return render_template('home.html')

@app.route('/detect')
def detect():
    """Fraud detection form page"""
    return render_template('detect.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Reload model if it wasn't loaded initially (e.g. if trained after app start)
    if not model:
        load_prediction_model()
        if not model:
            return render_template('result.html', prediction=-1) # Error state

    try:
        # Extract features from form
        type_val = request.form['type']
        
        # Helper function for safe float conversion
        def safe_float(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        amount = safe_float(request.form['amount'])
        oldbalanceOrg = safe_float(request.form['oldbalanceOrg'])
        newbalanceOrig = safe_float(request.form['newbalanceOrig'])
        oldbalanceDest = safe_float(request.form['oldbalanceDest'])
        newbalanceDest = safe_float(request.form['newbalanceDest'])

        # Mapping for 'type' (Must match training logic!)
        type_map = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
        type_encoded = type_map.get(type_val)
        
        # Validation
        if type_encoded is None:
             raise ValueError("Invalid transaction type")

        # Feature array: ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        features = np.array([[type_encoded, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]])
        
        # Debug: Print features and model info
        print(f"\n=== PREDICTION DEBUG ===")
        print(f"Input: type={type_val}({type_encoded}), amount={amount}")
        print(f"oldbalanceOrg={oldbalanceOrg}, newbalanceOrig={newbalanceOrig}")
        print(f"oldbalanceDest={oldbalanceDest}, newbalanceDest={newbalanceDest}")
        print(f"Feature array: {features}")
        
        # Check if model has feature_names attribute
        if hasattr(model, 'feature_names_in_'):
            print(f"Model expects features: {model.feature_names_in_}")
        if hasattr(model, 'n_features_in_'):
            print(f"Model expects {model.n_features_in_} features")
        
        # Predict
        prediction = model.predict(features)[0]
        print(f"Prediction result: {prediction}")
        print(f"========================\n")
        
        return render_template('result.html', prediction=int(prediction))

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('result.html', prediction=-1) # Handle error gracefully

if __name__ == '__main__':
    app.run(debug=True)
