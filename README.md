🏠 **Airbnb Price Prediction — GPU-Accelerated XGBoost Model**  
This project builds an end-to-end machine learning pipeline to predict Airbnb listing prices across multiple U.S. cities. It focuses on data cleaning, feature preprocessing, and a GPU-optimized XGBoost model for fast and accurate regression.

---

### 🚀 Key Features  
**Data Cleaning & Filtering**  
- Removed invalid or extreme prices, handled missing values, and normalized categorical data.  
- Filtered out unrealistic values (kept \$10–\$5000) to reduce outliers.  
- Merged low-frequency cities into a single “Other” category for efficiency.  

**Feature Engineering**  
- Scaled numeric features with `StandardScaler`.  
- Encoded categorical variables (`room_type`, `city`) via one-hot encoding.  
- Applied `log1p` transform on target (`price_log`) to stabilize variance.  

**Modeling**  
- Trained an **XGBoost Regressor** using GPU acceleration (`gpu_hist`).  
- Implemented early stopping to prevent overfitting and optimize iteration count.  
- Tuned depth, learning rate, and sampling ratios for balanced performance.  

**Evaluation**  
- Metrics: **RMSE**, **MAE**, **R²**  
- Visualizations: *Actual vs Predicted Prices* and *Residual Distribution*  

---

### 📊 Results Summary  
| Metric | Value | Description |  
|--------:|------:|:------------|  
| RMSE | **234.36** | Root Mean Squared Error — lower is better |  
| MAE | **87.33** | Mean Absolute Error in predicted prices |  
| R² | **0.332** | Coefficient of determination (model fit) |  

---

### 🧠 Tech Stack  
- **Language:** Python (3.10+)  
- **Libraries:** XGBoost, Scikit-learn, Pandas, NumPy, Seaborn, Matplotlib  
- **Hardware:** NVIDIA GPU (CUDA-accelerated training)  

---

### 🖼️ Visualizations  
- **Predicted vs Actual Prices:** demonstrates alignment between predictions and real values.  
- **Residual Histogram:** confirms balanced and unbiased model errors.  

---

### 💾 Output  
Final predictions are saved to **Airbnb_XGB_Results.csv**, containing:  
`ActualPrice` | `PredictedPrice`  


---

## 📚 Author

**Sina Firoozian** — Data Science & Machine Learning Enthusiast  
📧 [sina.firuzian@gmail.com]  
🌐 [https://github.com/firoozian]
