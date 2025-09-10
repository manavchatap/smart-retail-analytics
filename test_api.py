import pytest
import requests
import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_overview_analytics():
    """Test overview analytics endpoint"""
    response = client.get("/analytics/overview")
    assert response.status_code == 200
    data = response.json()
    assert "kpis" in data
    assert "monthly_trends" in data
    assert "top_products" in data
    assert "category_performance" in data

def test_sales_trends():
    """Test sales trends endpoint"""
    response = client.get("/analytics/sales-trends")
    assert response.status_code == 200
    data = response.json()
    assert "daily_trends" in data
    assert "regional_performance" in data
    assert "category_trends" in data

def test_sales_prediction():
    """Test sales prediction endpoint"""
    prediction_data = {
        "category": "Electronics",
        "month": 10,
        "year": 2025
    }
    response = client.post("/predict/sales", json=prediction_data)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_revenue" in data
    assert "confidence" in data
    assert data["category"] == "Electronics"

def test_customer_segments():
    """Test customer segments endpoint"""
    response = client.get("/customers/segments")
    assert response.status_code == 200
    data = response.json()
    assert "segment_analysis" in data
    assert "total_customers" in data

def test_customer_segment_prediction():
    """Test customer segment prediction"""
    customer_data = {
        "total_revenue": 5000.0,
        "purchase_frequency": 5,
        "recency": 30
    }
    response = client.post("/customers/predict-segment", json=customer_data)
    assert response.status_code == 200
    data = response.json()
    assert "segment_name" in data
    assert "segment_id" in data

def test_product_performance():
    """Test product performance endpoint"""
    response = client.get("/products/performance")
    assert response.status_code == 200
    data = response.json()
    assert "product_performance" in data
    assert "low_stock_alerts" in data

def test_forecast_analytics():
    """Test forecast analytics endpoint"""
    response = client.get("/analytics/forecast")
    assert response.status_code == 200
    data = response.json()
    assert "forecasts" in data
    assert len(data["forecasts"]) > 0

if __name__ == "__main__":
    # Run basic tests
    print("Running API tests...")

    try:
        test_root_endpoint()
        print("✅ Root endpoint test passed")

        test_health_check()
        print("✅ Health check test passed")

        test_overview_analytics()
        print("✅ Overview analytics test passed")

        print("🎉 All basic tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
