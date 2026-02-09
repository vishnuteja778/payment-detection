import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate():
    print("Loading model...")
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    print("Loading dataset sample...")
    # Load enough rows to get both classes
    df = pd.read_csv('../PS_20174392719_1491204439457_log.csv', nrows=200000)
    
    # Preprocess same as training
    type_map = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
    df['type'] = df['type'].map(type_map)
    
    # Drop columns
    cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'step']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Get all fraud samples found
    fraud = df[df['isFraud'] == 1]
    # Get equal amount of safe samples
    safe = df[df['isFraud'] == 0].sample(n=len(fraud), random_state=42)
    
    test_data = pd.concat([fraud, safe])
    
    X = test_data.drop('isFraud', axis=1)
    y_true = test_data['isFraud']
    
    print(f"\nEvaluating on {len(test_data)} samples ({len(fraud)} fraud, {len(safe)} safe)")
    
    y_pred = model.predict(X)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Show some failure cases
    test_data['predicted'] = y_pred
    failures = test_data[test_data['isFraud'] != test_data['predicted']]
    
    if not failures.empty:
        print(f"\nFound {len(failures)} failure cases. Top 5:")
        print(failures.head(5))
    else:
        print("\nNo failures found in this sample!")

if __name__ == "__main__":
    evaluate()
