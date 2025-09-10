# Smart Retail & Sales Analytics System

## Project Structure

```
smart-retail-analytics/
├── backend/
│   ├── main.py                     # FastAPI application
│   ├── requirements.txt            # Python dependencies
│   ├── Dockerfile                  # Docker container config
│   ├── retail_sales_data.csv       # Sample dataset
│   ├── customer_segments.csv       # Customer segmentation data
│   └── models/
│       ├── sales_forecast_model.pkl     # Trained ML model for sales forecasting
│       ├── customer_segmentation_model.pkl # Customer clustering model
│       ├── category_encoder.pkl          # Label encoder for categories
│       └── feature_scaler.pkl           # Feature scaler for clustering
├── frontend/
│   ├── index.html                  # Main dashboard HTML
│   ├── style.css                   # Styling and responsive design
│   └── app.js                      # JavaScript for interactivity and charts
└── README.md                       # Project documentation
```

## Features Implemented

### 1. Data Collection & Preparation
- Synthetic retail sales dataset (5000 records)
- Product categories: Electronics, Clothing, Food & Beverages, Home & Garden, Sports, Beauty
- Time series data with seasonal patterns
- Customer transaction history

### 2. Machine Learning Models
- **Sales Forecasting**: Random Forest Regressor for predicting future sales
- **Customer Segmentation**: K-Means clustering for customer analysis (VIP, Loyal, Active, At Risk)
- Model performance: R² Score of 0.25 for forecasting, 4 distinct customer segments

### 3. Backend API (FastAPI)
- RESTful endpoints for analytics and predictions
- Real-time sales forecasting
- Customer segmentation analysis
- Product performance metrics
- CORS enabled for frontend integration
- Swagger documentation at `/docs`

### 4. Frontend Dashboard (React-style with Vanilla JS)
- **Overview Page**: KPIs, trends, top products, category performance
- **Sales Analytics**: Regional performance, category trends, time series analysis
- **Forecasting**: Sales predictions, seasonal analysis, confidence intervals
- **Customer Segments**: Segmentation analysis, customer lifetime value
- **Products**: Product performance, inventory alerts, profit analysis

### 5. Data Visualization
- Interactive charts using Chart.js
- Line charts for trends and forecasting
- Bar charts for comparisons
- Pie charts for segments and categories
- Responsive design for all screen sizes

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /analytics/overview` - Overview analytics and KPIs
- `GET /analytics/sales-trends` - Detailed sales trends
- `POST /predict/sales` - Sales forecasting
- `GET /customers/segments` - Customer segmentation analysis
- `POST /customers/predict-segment` - Predict customer segment
- `GET /products/performance` - Product performance metrics
- `GET /analytics/forecast` - Multi-month sales forecasts

## Installation & Setup

### Backend Setup
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

3. Access API documentation: http://localhost:8000/docs

### Frontend Setup
1. Open `index.html` in a web browser
2. The dashboard will load with sample data
3. All charts and interactions are fully functional

### Docker Deployment
1. Build the Docker image:
   ```bash
   docker build -t smart-retail-analytics .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 smart-retail-analytics
   ```

## Key Features

✅ **Comprehensive Analytics Dashboard**
✅ **Machine Learning Sales Forecasting**
✅ **Customer Segmentation Analysis**
✅ **Real-time Data Visualization**
✅ **Responsive Web Design**
✅ **RESTful API Architecture**
✅ **Docker Containerization**
✅ **Interactive Charts & Filters**
✅ **Product Performance Tracking**
✅ **Inventory Management Alerts**

## Technologies Used

- **Backend**: FastAPI, Python, Scikit-learn, Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Machine Learning**: Random Forest, K-Means Clustering
- **Data**: Synthetic retail sales dataset
- **Containerization**: Docker
- **API Documentation**: Swagger/OpenAPI

This system provides a complete end-to-end solution for retail analytics with modern ML capabilities and professional dashboard interface.
