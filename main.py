from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from typing import List, Optional
import uvicorn
from pydantic import BaseModel
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Smart Retail Analytics API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models
try:
    sales_model = joblib.load('sales_forecast_model.pkl')
    category_encoder = joblib.load('category_encoder.pkl')
    segmentation_model = joblib.load('customer_segmentation_model.pkl')
    feature_scaler = joblib.load('feature_scaler.pkl')
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# Load data
try:
    df = pd.read_csv('retail_sales_data.csv')
    customer_segments = pd.read_csv('customer_segments.csv')
    print("✅ Data loaded successfully")
except Exception as e:
    print(f"❌ Error loading data: {e}")

# Pydantic models
class PredictionRequest(BaseModel):
    category: str
    month: int
    year: int = 2025

class CustomerSegmentRequest(BaseModel):
    total_revenue: float
    purchase_frequency: int
    recency: int

@app.get("/")
async def root():
    return {"message": "Smart Retail Analytics API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/analytics/overview")
async def get_overview_analytics():
    """Get overview analytics and KPIs"""
    try:
        # Calculate KPIs
        total_revenue = df['revenue'].sum()
        total_quantity = df['quantity'].sum()
        avg_order_value = df['revenue'].mean()
        customer_count = df['customer_id'].nunique()

        # Monthly trends
        df['date'] = pd.to_datetime(df['date'])
        monthly_sales = df.groupby(df['date'].dt.to_period('M')).agg({
            'revenue': 'sum',
            'quantity': 'sum'
        }).reset_index()
        monthly_sales['date'] = monthly_sales['date'].astype(str)

        # Top products
        top_products = df.groupby('product_name').agg({
            'revenue': 'sum',
            'quantity': 'sum'
        }).sort_values('revenue', ascending=False).head(10).reset_index()

        # Category performance
        category_performance = df.groupby('category').agg({
            'revenue': 'sum',
            'quantity': 'sum'
        }).reset_index()
        category_performance['percentage'] = (category_performance['revenue'] / total_revenue * 100).round(2)

        return {
            "kpis": {
                "total_revenue": float(total_revenue),
                "total_quantity": int(total_quantity),
                "avg_order_value": float(avg_order_value),
                "customer_count": int(customer_count)
            },
            "monthly_trends": monthly_sales.to_dict('records'),
            "top_products": top_products.to_dict('records'),
            "category_performance": category_performance.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/sales-trends")
async def get_sales_trends():
    """Get detailed sales trends by various dimensions"""
    try:
        df['date'] = pd.to_datetime(df['date'])

        # Daily trends
        daily_trends = df.groupby(df['date'].dt.date).agg({
            'revenue': 'sum',
            'quantity': 'sum'
        }).reset_index()
        daily_trends['date'] = daily_trends['date'].astype(str)

        # Regional performance
        regional_performance = df.groupby('region').agg({
            'revenue': 'sum',
            'quantity': 'sum'
        }).reset_index()

        # Category trends
        category_trends = df.groupby(['category', df['date'].dt.to_period('M')]).agg({
            'revenue': 'sum'
        }).reset_index()
        category_trends['date'] = category_trends['date'].astype(str)

        return {
            "daily_trends": daily_trends.to_dict('records'),
            "regional_performance": regional_performance.to_dict('records'),
            "category_trends": category_trends.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/sales")
async def predict_sales(request: PredictionRequest):
    """Predict sales for a given category and time period"""
    try:
        # Encode category
        category_encoded = category_encoder.transform([request.category])[0]

        # Create features for prediction
        features = np.array([[
            request.year,           # year
            request.month,          # month
            15,                     # day (mid-month)
            3,                      # dayofweek (Wednesday)
            (request.month-1)//3 + 1,  # quarter
            0,                      # is_weekend
            1 if request.month in [11, 12] else 0,  # is_holiday_season
            category_encoded        # category_encoded
        ]])

        # Make prediction
        prediction = sales_model.predict(features)[0]

        # Calculate confidence interval (simplified)
        confidence = 0.85 if request.month not in [11, 12] else 0.75

        return {
            "category": request.category,
            "month": request.month,
            "year": request.year,
            "predicted_revenue": float(prediction),
            "confidence": confidence,
            "lower_bound": float(prediction * 0.8),
            "upper_bound": float(prediction * 1.2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/customers/segments")
async def get_customer_segments():
    """Get customer segmentation analysis"""
    try:
        # Segment analysis
        segment_stats = customer_segments.groupby('segment').agg({
            'total_revenue': 'mean',
            'purchase_frequency': 'mean',
            'recency': 'mean',
            'customer_id': 'count'
        }).reset_index()

        segment_stats.columns = ['segment', 'avg_revenue', 'avg_frequency', 'avg_recency', 'customer_count']

        # Add segment labels
        segment_labels = {
            0: 'VIP Customers',
            1: 'At Risk Customers', 
            2: 'Active Customers',
            3: 'VIP Customers'
        }

        segment_stats['segment_name'] = segment_stats['segment'].map(segment_labels)

        return {
            "segment_analysis": segment_stats.to_dict('records'),
            "total_customers": len(customer_segments),
            "segment_distribution": customer_segments['segment'].value_counts().to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/customers/predict-segment")
async def predict_customer_segment(request: CustomerSegmentRequest):
    """Predict customer segment for given customer data"""
    try:
        # Prepare features
        features = np.array([[request.total_revenue, request.purchase_frequency, request.recency]])
        features_scaled = feature_scaler.transform(features)

        # Predict segment
        segment = segmentation_model.predict(features_scaled)[0]

        # Get segment name
        segment_labels = {
            0: 'VIP Customers',
            1: 'At Risk Customers',
            2: 'Active Customers', 
            3: 'VIP Customers'
        }

        return {
            "segment_id": int(segment),
            "segment_name": segment_labels.get(segment, 'Unknown'),
            "customer_data": {
                "total_revenue": request.total_revenue,
                "purchase_frequency": request.purchase_frequency,
                "recency": request.recency
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products/performance")
async def get_product_performance():
    """Get product performance metrics"""
    try:
        # Product performance
        product_stats = df.groupby(['product_name', 'category']).agg({
            'revenue': 'sum',
            'quantity': 'sum',
            'unit_price': 'mean'
        }).reset_index()

        # Calculate profit margin (simplified)
        product_stats['profit_margin'] = (product_stats['unit_price'] * 0.3).round(2)
        product_stats = product_stats.sort_values('revenue', ascending=False)

        # Low stock alerts (random for demo)
        low_stock_products = product_stats.sample(5)
        low_stock_products['stock_level'] = np.random.randint(5, 20, size=len(low_stock_products))

        return {
            "product_performance": product_stats.to_dict('records'),
            "low_stock_alerts": low_stock_products[['product_name', 'category', 'stock_level']].to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/forecast")
async def get_forecast_analytics():
    """Get sales forecasting data for next few months"""
    try:
        # Generate forecasts for next 4 months
        forecasts = []
        current_date = datetime.now()

        for i in range(1, 5):
            future_date = current_date + timedelta(days=30*i)
            month = future_date.month
            year = future_date.year

            # Predict for each category
            category_forecasts = []
            for category in df['category'].unique():
                try:
                    category_encoded = category_encoder.transform([category])[0]
                    features = np.array([[
                        year, month, 15, 3, (month-1)//3 + 1, 0,
                        1 if month in [11, 12] else 0, category_encoded
                    ]])
                    prediction = sales_model.predict(features)[0]
                    category_forecasts.append({
                        'category': category,
                        'predicted_revenue': float(prediction)
                    })
                except:
                    pass

            total_forecast = sum([cf['predicted_revenue'] for cf in category_forecasts])
            forecasts.append({
                'date': f"{year}-{month:02d}",
                'total_predicted_revenue': total_forecast,
                'confidence': 0.85 if month not in [11, 12] else 0.75,
                'category_breakdown': category_forecasts
            })

        return {"forecasts": forecasts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
