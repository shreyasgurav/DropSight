# 🛒 AI-Based User Journey & Drop-Off Prediction System

A Web Analytics AI project that predicts where users drop off on an e-commerce clothing website and provides actionable business recommendations.

## 📊 Project Overview

- **Dataset**: 165,474 browsing records from 24,026 unique user sessions
- **ML Model**: Random Forest Classifier with **98.7% accuracy**
- **Target**: Predict drop-off page (1-5) in user journey
- **Output**: Interactive Streamlit dashboard with predictions and insights

## 📁 Project Structure

```
Web analytics Mini Project/
├── app.py                      # Streamlit dashboard (main entry point)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/                        # Source code
│   ├── data_loader.py         # Data cleaning & feature engineering
│   ├── eda.py                 # Exploratory data analysis & charts
│   ├── model.py               # ML model training & evaluation
│   └── recommendations.py     # Business recommendation engine
│
├── data/                       # Data files
│   ├── e-shop clothing 2008.csv    # Raw dataset
│   └── cleaned_data.csv            # Processed dataset
│
└── output/                     # Generated artifacts
    ├── model.pkl              # Trained Random Forest model
    ├── model_metrics.csv      # Model comparison metrics
    └── charts/                # EDA visualizations (8 PNG files)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Pipeline
```bash
# Step 1: Clean and prepare data
python src/data_loader.py

# Step 2: Generate EDA charts
python src/eda.py

# Step 3: Train ML models
python src/model.py
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```

Open browser at: **http://localhost:8501**

## 📈 Model Performance

| Model | Accuracy | Status |
|-------|----------|--------|
| **Random Forest** | **98.70%** | ✅ Selected |
| Logistic Regression | 59.53% | ❌ Not used |

### Per-Page Performance
- Page 1: 99.4% correct predictions
- Page 2: 97.5% correct predictions
- Page 3: 97.6% correct predictions
- Page 4: 98.4% correct predictions
- Page 5: 99.8% correct predictions

## 🎯 Features

### Dashboard Pages

1. **📊 Overview**
   - Key metrics (sessions, records, countries)
   - Dataset preview

2. **📈 EDA & Insights**
   - 8 visualization charts with business insights
   - Drop-off analysis by page, category, country, price

3. **🔮 Drop-Off Predictor**
   - Interactive prediction form
   - Real-time confidence scores
   - Business recommendations
   - Visual user journey funnel

4. **🏆 Model Performance**
   - Accuracy comparison
   - Confusion matrix
   - Feature importance
   - Classification report

## 💡 Business Recommendations

The system provides page-specific recommendations:

- **Page 1**: Improve homepage layout, add trending products
- **Page 2**: Add better filters and navigation
- **Page 3**: Show discount popup or stock alerts
- **Page 4**: Offer free shipping, show reviews
- **Page 5**: Send cart abandonment emails

Plus price-aware suggestions for high-priced items.

## 🛠️ Tech Stack

- **Python 3.12+**
- **Data**: Pandas, NumPy
- **ML**: Scikit-learn (Random Forest)
- **Visualization**: Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Model Persistence**: Joblib

## 📊 Dataset Details

- **Rows**: 165,474
- **Columns**: 15
- **Features**: category, country, price, color, location, photo type, order step, month
- **Target**: page (1-5)
- **Engineered Feature**: dropped_off (binary)

## ✅ Validation Results

All validation checks passed:
- ✅ No missing values
- ✅ Feature engineering logic correct
- ✅ Charts represent data accurately
- ✅ Model generalizes well (no overfitting)
- ✅ Predictions are logical and consistent

**Status**: Production-ready ML model

## 📝 Notes

- Model retraining recommended quarterly to adapt to changing user behavior
- All charts auto-generated in `output/charts/`
- Dashboard uses caching for optimal performance
