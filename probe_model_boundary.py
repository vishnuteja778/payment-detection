import pickle
import numpy as np

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Probing model decision boundary for TRANSFER transactions...")
print("Scenario: Sender balance is fully drained (OldBalance = Amount, NewBalance = 0)")

amounts = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]

for amt in amounts:
    # Feature order: type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
    # Type 4 = TRANSFER
    features = [[4, amt, amt, 0, 0, 0]]
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    
    status = "FRAUD" if pred == 1 else "SAFE"
    print(f"Amount: ${amt:<10} -> Prediction: {status} (Prob Fraud: {prob[1]:.4f})")

print("\nProbing for CASH_OUT...")
for amt in amounts:
    # Type 1 = CASH_OUT
    features = [[1, amt, amt, 0, 0, amt]] # Receiver gets it
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    
    status = "FRAUD" if pred == 1 else "SAFE"
    print(f"Amount: ${amt:<10} -> Prediction: {status} (Prob Fraud: {prob[1]:.4f})")
