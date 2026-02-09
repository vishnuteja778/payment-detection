import requests

# Test the Flask prediction endpoint
url = "http://127.0.0.1:5000/predict"

# Fraud test case
data = {
    "type": "TRANSFER",
    "amount": 200000,
    "oldbalanceOrg": 200000,
    "newbalanceOrig": 0,
    "oldbalanceDest": 0,
    "newbalanceDest": 200000
}

print("Testing prediction endpoint...")
print(f"Data: {data}")

try:
    response = requests.post(url, data=data)
    print(f"Status: {response.status_code}")
    
    # Check if fraud was detected in the response
    if "Fraud Detected" in response.text:
        print("✅ RESULT: Fraud detected correctly!")
    elif "Transaction Verified" in response.text or "Transaction Safe" in response.text:
        print("❌ RESULT: Transaction marked as SAFE (should be fraud)")
    elif "Error" in response.text:
        print("⚠️ RESULT: Error occurred")
    else:
        print("❓ RESULT: Unknown response")
        
except Exception as e:
    print(f"Error: {e}")
