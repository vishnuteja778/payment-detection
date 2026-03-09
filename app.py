from flask import Flask, render_template, request, send_file, redirect, url_for
import pickle
import numpy as np
import os
import json
import sqlite3
from datetime import datetime
from fpdf import FPDF
import io
import tempfile

app = Flask(__name__)

# ─── Model loading ───
model_path = 'model.pkl'
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

# ─── Database setup ───
DB_PATH = 'transactions.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        type TEXT NOT NULL,
        amount REAL NOT NULL,
        oldbalanceOrg REAL NOT NULL,
        newbalanceOrig REAL NOT NULL,
        oldbalanceDest REAL NOT NULL,
        newbalanceDest REAL NOT NULL,
        prediction INTEGER NOT NULL,
        confidence REAL NOT NULL
    )''')
    conn.commit()
    conn.close()

init_db()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ─── Helper: Load JSON data ───
def load_json(filepath, default=None):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return default or {}

# ─── Type mapping ───
TYPE_MAP = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
TYPE_LABELS = {v: k for k, v in TYPE_MAP.items()}

# ─── Routes ───

@app.route('/')
def home():
    """Landing page with platform overview"""
    return render_template('home.html')

@app.route('/detect')
def detect():
    """Fraud detection form page"""
    return render_template('detect.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        load_prediction_model()
        if not model:
            return render_template('result.html', prediction=-1, confidence=0)

    try:
        # Extract features
        type_val = request.form['type']

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

        type_encoded = TYPE_MAP.get(type_val)
        if type_encoded is None:
            raise ValueError("Invalid transaction type")

        features = np.array([[type_encoded, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]])

        # Predict with confidence
        prediction = int(model.predict(features)[0])

        confidence = 0.0
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = round(float(proba[prediction]) * 100, 1)
        else:
            confidence = 100.0

        print(f"\n=== PREDICTION ===")
        print(f"Type={type_val}, Amount={amount}, Prediction={prediction}, Confidence={confidence}%")

        # Save to history
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO transactions
            (timestamp, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, prediction, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (timestamp, type_val, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, prediction, confidence))
        conn.commit()
        txn_id = cursor.lastrowid
        conn.close()

        return render_template('result.html',
                             prediction=prediction,
                             confidence=confidence,
                             txn_id=txn_id,
                             txn_type=type_val,
                             amount=amount)

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return render_template('result.html', prediction=-1, confidence=0)


@app.route('/history')
def history():
    """Transaction history page"""
    conn = get_db()
    transactions = conn.execute(
        'SELECT * FROM transactions ORDER BY id DESC LIMIT 50'
    ).fetchall()
    conn.close()
    return render_template('history.html', transactions=transactions)


@app.route('/dashboard')
def dashboard():
    """Model comparison dashboard"""
    metrics = load_json('model_metrics.json')
    importance = load_json('feature_importance.json')
    cm = load_json('confusion_matrix.json')

    # Get feature importances from the actual model if available
    if model and hasattr(model, 'feature_importances_'):
        feature_names = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        importances = model.feature_importances_.tolist()
        importance = {'features': feature_names, 'importances': importances}

    return render_template('dashboard.html',
                         metrics=metrics,
                         importance=importance,
                         confusion_matrix=cm)


@app.route('/report/<int:txn_id>')
def report(txn_id):
    """Generate PDF report for a transaction"""
    conn = get_db()
    txn = conn.execute('SELECT * FROM transactions WHERE id = ?', (txn_id,)).fetchone()
    conn.close()

    if not txn:
        return "Transaction not found", 404

    # Build PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(26, 86, 219)
    pdf.cell(0, 15, 'FraudGuard AI', new_x="LMARGIN", new_y="NEXT", align='C')

    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, 'Transaction Analysis Report', new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(5)

    # Divider
    pdf.set_draw_color(26, 86, 219)
    pdf.set_line_width(0.8)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(10)

    # Report metadata
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, f'Report ID: TXN-{txn["id"]:05d}', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f'Generated: {txn["timestamp"]}', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Result banner
    is_fraud = txn['prediction'] == 1
    if is_fraud:
        pdf.set_fill_color(254, 226, 226)
        pdf.set_text_color(185, 28, 28)
        result_text = 'FRAUD DETECTED'
    else:
        pdf.set_fill_color(209, 250, 229)
        pdf.set_text_color(6, 95, 70)
        result_text = 'TRANSACTION SAFE'

    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 14, result_text, new_x="LMARGIN", new_y="NEXT", align='C', fill=True)
    pdf.ln(3)

    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, f'Confidence: {txn["confidence"]}%', new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(10)

    # Transaction details table
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(26, 86, 219)
    pdf.cell(0, 8, 'Transaction Details', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    details = [
        ('Transaction Type', txn['type']),
        ('Amount', f'${txn["amount"]:,.2f}'),
        ('Sender Old Balance', f'${txn["oldbalanceOrg"]:,.2f}'),
        ('Sender New Balance', f'${txn["newbalanceOrig"]:,.2f}'),
        ('Receiver Old Balance', f'${txn["oldbalanceDest"]:,.2f}'),
        ('Receiver New Balance', f'${txn["newbalanceDest"]:,.2f}'),
    ]

    for i, (label, value) in enumerate(details):
        if i % 2 == 0:
            pdf.set_fill_color(245, 247, 250)
        else:
            pdf.set_fill_color(255, 255, 255)

        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(80, 9, f'  {label}', fill=True)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 9, value, new_x="LMARGIN", new_y="NEXT", fill=True)

    pdf.ln(12)

    # Model info
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(26, 86, 219)
    pdf.cell(0, 8, 'Analysis Method', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 6,
        'This analysis was performed using a Random Forest Classifier trained on 6.3 million '
        'real financial transactions. The model uses ensemble learning with class-weight balancing '
        'to detect fraudulent patterns with high precision and recall.')
    pdf.ln(10)

    # Footer
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.3)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(5)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, 'This report was generated automatically by FraudGuard AI. For educational purposes only.', 
             new_x="LMARGIN", new_y="NEXT", align='C')

    # Output to bytes
    pdf_bytes = pdf.output()

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'FraudGuard_Report_TXN-{txn_id:05d}.pdf'
    )


if __name__ == '__main__':
    app.run(debug=True)
