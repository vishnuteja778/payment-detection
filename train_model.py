import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os

# Set style for plots
sns.set(style="whitegrid")

def load_data(filepath):
    """Loads the dataset from the specified filepath."""
    if not os.path.exists(filepath):
        # Try checking the parent directory (Desktop)
        parent_path = os.path.join(os.path.dirname(os.getcwd()), filepath)
        if os.path.exists(parent_path):
            print(f"Found dataset in parent directory: {parent_path}")
            return pd.read_csv(parent_path)
            
        print(f"Error: File '{filepath}' not found in current or parent directory.")
        return None
    print(f"Loading dataset from {filepath}...")
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Performs data preprocessing: cleaning, encoding, and splitting."""
    print("Preprocessing data...")
    
    # Check for missing values
    if df.isnull().sum().any():
        print("Handling missing values...")
        df = df.dropna()

    # Drop unnecessary columns
    cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'step']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Encode 'type' column
    type_map = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
    df['type'] = df['type'].map(type_map)
    
    # IMPORTANT: Fraud only occurs in CASH_OUT (1) and TRANSFER (4) types
    # We can focus training on relevant data
    
    # Feature selection
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']
    
    print(f"Total samples: {len(X)}, Fraud cases: {y.sum()} ({y.mean()*100:.4f}%)")
    
    # Train-test split (stratify to maintain class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle class imbalance using undersampling of majority class
    # This is more practical than SMOTE for very large datasets
    print("Balancing training data...")
    
    # Separate majority and minority
    X_train_combined = pd.concat([X_train, y_train], axis=1)
    fraud_samples = X_train_combined[X_train_combined['isFraud'] == 1]
    non_fraud_samples = X_train_combined[X_train_combined['isFraud'] == 0]
    
    # Undersample: take equal fraud samples from non-fraud (1:1 ratio)
    # CRITICAL FIX: We must sample from types that behave like fraud (TRANSFER/CASH_OUT)
    # otherwise the model just learns "TRANSFER = Fraud" because safe samples are mostly PAYMENTs.
    # We increase ratio to 1:4 (Fraud:Safe) to reduce False Positives
    n_fraud = len(fraud_samples)
    n_undersample = n_fraud * 4
    
    # Filter non-fraud to only relevant types (1=CASH_OUT, 4=TRANSFER)
    relevant_types = [1, 4]
    non_fraud_hard = non_fraud_samples[non_fraud_samples['type'].isin(relevant_types)]
    non_fraud_easy = non_fraud_samples[~non_fraud_samples['type'].isin(relevant_types)]
    
    # We want a mix: 
    # 1. Hard negatives (Safe Transfers): to prevent False Positives on Transfers
    # 2. Easy negatives (Payments): to ensure they are always Safe
    
    n_safe_hard = int(n_undersample * 0.5)
    n_safe_easy = n_undersample - n_safe_hard
    
    print(f"Sampling {n_safe_hard} hard negatives and {n_safe_easy} easy negatives...")
    
    sample_hard = non_fraud_hard.sample(n=min(n_safe_hard, len(non_fraud_hard)), random_state=42)
    sample_easy = non_fraud_easy.sample(n=min(n_safe_easy, len(non_fraud_easy)), random_state=42)
    
    non_fraud_undersampled = pd.concat([sample_hard, sample_easy])
    
    # Combine
    balanced_train = pd.concat([fraud_samples, non_fraud_undersampled])
    balanced_train = balanced_train.sample(frac=1, random_state=42)  # Shuffle
    
    X_train_balanced = balanced_train.drop('isFraud', axis=1)
    y_train_balanced = balanced_train['isFraud']
    
    print(f"Balanced training set: {len(X_train_balanced)} samples")
    print(f"  Fraud: {y_train_balanced.sum()}, Non-fraud: {len(y_train_balanced) - y_train_balanced.sum()}")
    
    return X_train_balanced, X_test, y_train_balanced, y_test

def perform_eda(df):
    """Performs Exploratory Data Analysis and saves plots."""
    print("Performing EDA...")
    
    if not os.path.exists('static'):
        os.makedirs('static')

    # Countplot of Transaction Types
    plt.figure(figsize=(10,6))
    sns.countplot(x='type', data=df)
    plt.title('Distribution of Transaction Types')
    plt.savefig('static/transaction_types.png')
    plt.close()
    
    # Countplot of Fraud vs Non-Fraud
    plt.figure(figsize=(6,4))
    sns.countplot(x='isFraud', data=df)
    plt.title('Fraud vs Legitimate Transactions')
    plt.savefig('static/fraud_counts.png')
    plt.close()
    
    print("EDA plots saved to 'static/' directory.")

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Trains multiple models and evaluates them."""
    
    # Use class_weight='balanced' to handle imbalanced fraud data
    # Fraud cases are typically <1% of all transactions
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=50, class_weight='balanced')
    }
    
    best_model = None
    best_f1 = 0
    
    results = {}
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
        print(f"{name} Results: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    print("\nBest Model selected based on F1-Score.")
    return best_model, results

def save_model(model, filename='model.pkl'):
    """Saves the trained model to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

if __name__ == "__main__":
    DATASET_PATH = "PS_20174392719_1491204439457_log.csv"
    
    # Main workflow
    df = load_data(DATASET_PATH)
    
    if df is not None:
        # Exploratory Data Analysis
        # perform_eda(df) # Commented out to save time if running repeatedly, uncomment for full run
        perform_eda(df)

        # Preprocessing
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # Model Training
        best_model, _ = train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Save Model
        if best_model:
            save_model(best_model)
