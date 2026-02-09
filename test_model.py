import pickle
import numpy as np

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print("=" * 50)
print("MODEL DEBUG TEST")
print("=" * 50)
print(f"Model type: {type(model).__name__}")
print(f"Expected features: {model.n_features_in_}")

# Type mapping: CASH_IN=0, CASH_OUT=1, DEBIT=2, PAYMENT=3, TRANSFER=4
# Feature order: type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest

print("\n--- Test Cases ---")

# Test 1: Typical fraud pattern (TRANSFER, large amount, balance fully drained)
test1 = np.array([[4, 1000000, 1000000, 0, 0, 1000000]])  # TRANSFER
pred1 = model.predict(test1)[0]
print(f"Test 1 - TRANSFER, 1M, balance drained: {pred1} {'(FRAUD)' if pred1 == 1 else '(SAFE)'}")

# Test 2: CASH_OUT fraud pattern
test2 = np.array([[1, 500000, 500000, 0, 0, 500000]])  # CASH_OUT
pred2 = model.predict(test2)[0]
print(f"Test 2 - CASH_OUT, 500K, balance drained: {pred2} {'(FRAUD)' if pred2 == 1 else '(SAFE)'}")

# Test 3: Normal payment
test3 = np.array([[3, 150, 500, 350, 0, 150]])  # PAYMENT
pred3 = model.predict(test3)[0]
print(f"Test 3 - PAYMENT, $150, normal: {pred3} {'(FRAUD)' if pred3 == 1 else '(SAFE)'}")

# Test 4: Extreme fraud pattern
test4 = np.array([[4, 10000000, 10000000, 0, 0, 0]])  # TRANSFER, huge amount
pred4 = model.predict(test4)[0]
print(f"Test 4 - TRANSFER, 10M, extreme: {pred4} {'(FRAUD)' if pred4 == 1 else '(SAFE)'}")

# Test 5: Another fraud pattern - recipient receives but had 0 balance
test5 = np.array([[1, 200000, 200000, 0, 0, 200000]])  # CASH_OUT
pred5 = model.predict(test5)[0]
print(f"Test 5 - CASH_OUT, 200K: {pred5} {'(FRAUD)' if pred5 == 1 else '(SAFE)'}")

print("\n--- Summary ---")
fraud_count = sum([pred1, pred2, pred4, pred5])
print(f"Fraud cases detected: {fraud_count}/4 suspicious transactions")
print(f"Safe case correct: {'Yes' if pred3 == 0 else 'No'}")
