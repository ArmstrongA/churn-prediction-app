from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_endpoint():
    data = {
        "Gender": "Male",
        "Senior Citizen": "Yes",
        "Partner": "Yes",
        "Dependents": "Yes",
        "Tenure Months": 68,
        "Phone Service": "Yes",
        "Multiple Lines": "Yes",
        "Internet Service": "DSL",
        "Online Security": "Yes",
        "Online Backup": "Yes",
        "Device Protection": "Yes",
        "Tech Support": "Yes",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Mailed check",
        "Monthly Charges": 56.7,
        "Total Charges": 456.5
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "churn_probability" in response.json()
