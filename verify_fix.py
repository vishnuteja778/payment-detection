import pickle
import numpy as np
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore")

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print("="*60)
print("FINAL MODEL VERIFICATION")
print("="*60)

# Test 1: Partial Transfer to NEW Account (Dest=0)
# This mimics "New" account creation which might be suspicious
test1 = [[4, 1000, 5000, 4000, 0, 1000]]
pred1 = model.predict(test1)[0]
print(f"1. Partial Transfer to NEW account (Dest=0): {'FRAUD ' if pred1==1 else 'SAFE '}")

# Test 1.1: Partial Transfer to EXISTING Account (Dest=500)
# This mimics "Paying a friend" or standard transfer
test1_1 = [[4, 1000, 5000, 4000, 500, 1500]]
pred1_1 = model.predict(test1_1)[0]
print(f"1.1 Partial Transfer to EXISTING account (Dest=500): {'FRAUD ❌' if pred1_1==1 else 'SAFE ✅'}")

# Test 2: Full Drain Transfer (Should be FRAUD)
test2 = [[4, 5000, 5000, 0, 0, 0]]
pred2 = model.predict(test2)[0]
print(f"2. Full Drain Transfer ($5000 from $5000): {'FRAUD ✅' if pred2==1 else 'SAFE ❌'}")

# Test 3: Standard Payment (Should be SAFE)
test3 = [[3, 100, 500, 400, 0, 0]]
pred3 = model.predict(test3)[0]
print(f"3. Standard Payment ($100): {'FRAUD ❌' if pred3==1 else 'SAFE ✅'}")

# Test 4: High Value CashOut (Should be FRAUD)
test4 = [[1, 200000, 200000, 0, 0, 200000]]
pred4 = model.predict(test4)[0]
print(f"4. High Value CashOut ($200k drain): {'FRAUD ✅' if pred4==1 else 'SAFE ❌'}")

print("="*60)
