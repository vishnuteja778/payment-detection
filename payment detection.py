from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Warning: model.pkl not found. Please train the model first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('result.html', prediction=-1) # Error state

    try:
        # Extract features from form
        # Note: We need to match the exact feature order and preprocessing as during training
        # This is a placeholder logic based on PaySim columns usually used
        
        # Correct mapping based on alphabetical LabelEncoder
        type_map = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
        
        type_val = request.form['type']
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        # Simple encoding - this MUST match train_model.py logic
        type_encoded = type_map.get(type_val, 0)

        # Feature array
        features = np.array([[type_encoded, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]])
        
        # Predict
        prediction = model.predict(features)[0]
        
        return render_template('result.html', prediction=prediction)

    except Exception as e:
        print(f"Error: {e}")
        return render_template('result.html', prediction=-1) # Handle error gracefully in template if needed

if __name__ == '__main__':
    app.run(debug=True)
