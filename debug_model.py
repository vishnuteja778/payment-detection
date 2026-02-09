"""
Debug script to understand why model isn't detecting fraud
"""
import pickle
import numpy as np
import pandas as pd
import os

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print("=" * 60)
print("MODEL ANALYSIS")
print("=" * 60)
print(f"Model type: {type(model).__name__}")
print(f"Features expected: {model.n_features_in_}")
if hasattr(model, 'feature_names_in_'):
    print(f"Feature names: {list(model.feature_names_in_)}")

# Load some actual fraud data to understand patterns
dataset_path = "../PS_20174392719_1491204439457_log.csv"
if os.path.exists(dataset_path):
    print("\n" + "=" * 60)
    print("ANALYZING ACTUAL FRAUD DATA")
    print("=" * 60)
    
    df = pd.read_csv(dataset_path)
    fraud_df = df[df['isFraud'] == 1]
    
    # Type mapping
    type_map = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
    
    # Get sample fraud transactions
    print("\nSample fraud transactions:")
    sample_fraud = fraud_df.head(5)
    for idx, row in sample_fraud.iterrows():
        type_encoded = type_map[row['type']]
        features = [[type_encoded, row['amount'], row['oldbalanceOrg'], 
                    row['newbalanceOrig'], row['oldbalanceDest'], row['newbalanceDest']]]
        pred = model.predict(features)[0]
        print(f"  Type={row['type']}({type_encoded}), Amount={row['amount']:.0f}")
        print(f"    BalOrg: {row['oldbalanceOrg']:.0f} -> {row['newbalanceOrig']:.0f}")
        print(f"    BalDest: {row['oldbalanceDest']:.0f} -> {row['newbalanceDest']:.0f}")
        print(f"    Model Prediction: {pred} {'(FRAUD)' if pred == 1 else '(SAFE)'}")
        print()

print("=" * 60)
print("TESTING MANUAL CASES")
print("=" * 60)

# Feature order: type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
test_cases = [
    ("TRANSFER fraud", [4, 200000, 200000, 0, 0, 200000]),
    ("CASH_OUT fraud", [1, 500000, 500000, 0, 0, 500000]),
    ("PAYMENT safe", [3, 150, 500, 350, 0, 150]),
]

for name, features in test_cases:
    pred = model.predict([features])[0]
    prob = model.predict_proba([features])[0] if hasattr(model, 'predict_proba') else None
    print(f"{name}: {pred} {'(FRAUD)' if pred == 1 else '(SAFE)'}")
    if prob is not None:
        print(f"  Probability: P(safe)={prob[0]:.4f}, P(fraud)={prob[1]:.4f}")
