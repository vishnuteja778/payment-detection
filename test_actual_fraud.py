import pickle
import pandas as pd
import numpy as np

m = pickle.load(open('model.pkl', 'rb'))

# Load actual fraud data
df = pd.read_csv('../PS_20174392719_1491204439457_log.csv', nrows=100000)
type_map = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
df['type_encoded'] = df['type'].map(type_map)

fraud = df[df['isFraud']==1].head(5)
print('Testing 5 actual fraud transactions from dataset:')
correct = 0
for i, row in fraud.iterrows():
    X = [[row['type_encoded'], row['amount'], row['oldbalanceOrg'], row['newbalanceOrig'], row['oldbalanceDest'], row['newbalanceDest']]]
    pred = m.predict(X)[0]
    ttype = row['type']
    print(f'{ttype}: amount={row["amount"]:.0f}, pred={pred}')
    if pred == 1:
        correct += 1

print(f'\nCorrect fraud detections: {correct}/5')
