# 🏠 Airbnb Price Prediction — GPU-Accelerated XGBoost Model

This project builds an **end-to-end machine learning pipeline** to predict Airbnb listing prices across multiple U.S. cities.  
It focuses on data cleaning, feature preprocessing, and a GPU-optimized XGBoost model for fast and accurate regression.

---

## 🚀 Key Features

- **Data Cleaning & Filtering**: Removed invalid or extreme prices, handled missing values, and normalized text/categorical data.  
- **Feature Engineering**:  
  - Scaled numeric features using `StandardScaler`  
  - Encoded categorical variables (`room_type`, `city`) via `OneHotEncoder`  
  - Applied `log1p` transform on target (`price_log`) to stabilize variance  
- **Modeling**:  
  - Trained an **XGBoost Regressor** with GPU acceleration (`gpu_hist`)  
  - Used **Early Stopping** to prevent overfitting and optimize iterations  
- **Evaluation**:  
  - Metrics: `RMSE`, `MAE`, `R²`  
  - Visualizations: *Actual vs Predicted Prices* and *Residual Distribution*  
- **Performance**:  
  - RMSE ≈ **234.36**  
  - MAE ≈ **87.33**  
  - R² ≈ **0.332**

---

## 🧠 Tech Stack

- **Language:** Python (3.11+)  
- **Libraries:** XGBoost, Scikit-learn, Pandas, NumPy, Seaborn, Matplotlib  
- **Hardware:** NVIDIA GPU (CUDA-accelerated training)

---

## 📊 Results Summary

| Metric | Value | Description |
|--------|--------|-------------|
| RMSE | 234.36 | Root Mean Squared Error — lower is better |
| MAE | 87.33 | Mean Absolute Error in predicted prices |
| R² | 0.332 | Coefficient of determination (model fit) |

---

## 🖼️ Sample Visualizations

- **Predicted vs Actual Prices** → shows overall prediction accuracy  
- **Residual Histogram** → confirms balanced and unbiased model errors  

---

## 💾 Output

The final predictions are saved in **`Airbnb_XGB_Results.csv`**, containing the columns:  
`ActualPrice` and `PredictedPrice`.

---

## 📚 Author

**Sina Firoozian** — Data Science & Machine Learning Enthusiast  
📧 [sina.firuzian@gmail.com]  
🌐 [https://github.com/firoozian]
