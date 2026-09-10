# Food Delivery Demand Forecasting Using LSTM

## Overview

Accurate demand forecasting is important for food delivery businesses to maintain sufficient inventory, optimize fulfilment-center operations, and allocate resources efficiently.

This project develops an end-to-end **weekly food demand forecasting system using Long Short-Term Memory (LSTM) neural networks**. Historical order data is analyzed to identify demand patterns across meals and fulfilment centers, followed by the development and comparison of multiple forecasting approaches.

The final model is a **multivariate LSTM with a log-transformed demand target**, using historical demand, pricing, and promotional features to forecast future weekly meal demand.

---

## Problem Statement

A food delivery company needs to forecast weekly demand for different meals across its fulfilment centers.

The objective is to use historical demand patterns along with pricing, promotional, meal, and fulfilment-center information to predict future orders and support:

- Inventory planning
- Resource allocation
- Fulfilment-center operations
- Meal-level demand planning
- Promotional decision-making

---

## Objectives

### 1. Understand Demand Patterns

Analyze historical demand across:

- Weekly demand trends
- Meal popularity
- Fulfilment centers
- Meal categories
- Cuisines
- Pricing
- Promotional campaigns

### 2. Perform Time-Series Analysis

Investigate the temporal structure of demand using:

- Weekly demand aggregation
- Historical demand trends
- Autocorrelation
- Partial autocorrelation
- Center-meal level time series

### 3. Build Forecasting Models

Develop and compare multiple forecasting approaches:

```text
Historical Demand
       ↓
Previous-Week Baseline
       ↓
Raw-Target LSTM
       ↓
Log-Transformed LSTM
       ↓
Multivariate LSTM
```

### 4. Forecast Future Demand

Use the selected LSTM model to forecast weekly demand for future meal-center combinations.

---

## Dataset

The dataset contains historical food delivery orders along with information about meals and fulfilment centers.

### Dataset Files

| File | Description |
|---|---|
| `train.csv` | Historical meal demand and related features |
| `test.csv` | Future observations for forecasting |
| `meal_info.csv` | Meal category and cuisine information |
| `fulfilment_center_info.csv` | Fulfilment-center information |
| `sample_submission.csv` | Required submission format |

### Important Features

| Feature | Description |
|---|---|
| `id` | Unique record identifier |
| `week` | Week number |
| `center_id` | Fulfilment center identifier |
| `meal_id` | Meal identifier |
| `checkout_price` | Price at checkout |
| `base_price` | Base meal price |
| `emailer_for_promotion` | Email promotion indicator |
| `homepage_featured` | Homepage promotion indicator |
| `num_orders` | Number of orders — target variable |

Additional meal and fulfilment-center information is obtained by joining the meal and center datasets with the order data.

---

## Project Workflow

```text
Raw Data
   ↓
Data Cleaning & Integration
   ↓
Exploratory Data Analysis
   ↓
Time-Series Analysis
   ↓
Complete Center-Meal Series Selection
   ↓
Chronological Train / Validation / Test Split
   ↓
Feature Scaling & Transformation
   ↓
Previous-Week Baseline
   ↓
LSTM Experiments
   ├── V1: Raw Target LSTM
   ├── V2: Log-Transformed Target LSTM
   └── V3: Multivariate LSTM
   ↓
Model Evaluation
   ↓
Final Model Selection
   ↓
Recursive Future Forecasting
   ↓
Final Submission
```

---

## Exploratory Data Analysis

The exploratory analysis investigates the structure and behavior of food delivery demand.

The analysis includes:

- Overall demand distribution
- Weekly demand trends
- Meal-level demand
- Center-wise demand
- Category-wise demand
- Cuisine-wise demand
- Price analysis
- Promotional impact
- Demand variability
- Time-series autocorrelation
- Partial autocorrelation

The target variable is highly right-skewed, with a small number of observations containing very high order volumes. This behavior was considered during model development.

---

## Time-Series Preparation

The dataset contains multiple time series corresponding to different **center-meal combinations**.

To maintain consistent sequential input for the initial LSTM experiments, center-meal combinations with complete historical weekly observations were selected.

The data was divided chronologically to prevent future information from leaking into model training.

```text
Training     → Weeks 1–101
Validation   → Weeks 102–123
Holdout Test → Weeks 124–145
Future Test  → Weeks 146–155
```

An **8-week lookback window** was used for the LSTM models.

Each training sample therefore contains the previous 8 weeks of observations to predict the following week's demand.

---

## Target Transformation

The demand variable `num_orders` is highly skewed and contains extreme demand spikes.

To reduce the influence of these extreme values, the LSTM experiments use:

```python
log1p(num_orders)
```

After prediction, the output is converted back to the original demand scale using:

```python
expm1()
```

This transformation provides a more stable target distribution for neural-network training.

---

# Model Development

## 1. Previous-Week Baseline

A simple baseline model was developed before training the LSTM models.

The baseline predicts the current week's demand using the demand observed in the previous week.

This provides a reference point for evaluating whether the neural-network models provide meaningful improvement.

---

## 2. V1 — Raw Target LSTM

The first LSTM model directly predicts the scaled `num_orders` target.

### Architecture

```text
Input Sequence
      ↓
LSTM — 64 Units
      ↓
Dropout — 0.2
      ↓
Dense — 32 Units, ReLU
      ↓
Dense — 1 Unit
```

The model uses:

- Adam optimizer
- Mean Squared Error loss
- Mean Absolute Error as an additional metric
- Early stopping
- 8-week lookback window

---

## 3. V2 — Log-Transformed Target LSTM

The second model uses a `log1p` transformation on the target variable before scaling and training.

This approach was introduced to reduce the effect of extreme demand values.

### Architecture

```text
8-Week Demand Sequence
          ↓
     LSTM — 64
          ↓
     Dropout — 0.2
          ↓
     Dense — 32
          ↓
      Dense — 1
```

The predicted values are converted back to the original order scale using the inverse logarithmic transformation.

---

## 4. V3 — Multivariate LSTM

The final model extends the log-transformed LSTM by incorporating additional demand-related features.

### Input Features

```text
Historical Orders
Checkout Price
Base Price
Email Promotion
Homepage Featured
```

The model therefore uses both historical demand and external demand-related variables to predict future orders.

### Architecture

```text
8-Week Multivariate Sequence
             ↓
        LSTM — 64
             ↓
        Dropout — 0.2
             ↓
        Dense — 32
             ↓
         Dense — 1
```

The final model uses:

- 8-week lookback window
- 5 input features
- LSTM with 64 units
- Dropout regularization
- Dense layer with 32 neurons
- ReLU activation
- Adam optimizer
- Early stopping

---

# Model Evaluation

The models were evaluated using two primary regression metrics.

## Mean Absolute Error — MAE

MAE measures the average absolute difference between actual and predicted demand.

Lower MAE indicates better average forecasting accuracy.

## Root Mean Squared Error — RMSE

RMSE gives greater weight to larger errors and is therefore useful for evaluating the model's performance on high-demand observations.

Lower RMSE indicates fewer large prediction errors.

---

## Results

| Model | MAE | RMSE |
|---|---:|---:|
| Previous-Week Baseline | 158.78 | 383.11 |
| V1 — Raw Target LSTM | 137.69 | 329.34 |
| V2 — Log Target LSTM | 136.18 | 330.20 |
| **V3 — Multivariate LSTM** | **134.13** | 335.70 |

### Performance Comparison

The final V3 model achieved the lowest MAE among the evaluated models.

Compared with the previous-week baseline:

- MAE improved from **158.78 to 134.13**
- RMSE improved from **383.11 to 335.70**
- MAE improvement was approximately **15.5%**
- RMSE improvement was approximately **12.4%**

The results demonstrate that the LSTM models provide a meaningful improvement over a simple previous-week forecasting strategy.

---

## Key Findings

### Demand is highly skewed

Food delivery demand contains significant variation and extreme order spikes. The `log1p` transformation helped create a more stable target distribution for LSTM training.

### Recent history is informative

The previous week's demand provides a useful forecasting signal, which is reflected in the performance of the baseline model.

### LSTM improves over the baseline

Both the raw-target and log-transformed LSTM models outperform the previous-week baseline on MAE and RMSE.

### Additional features provide further improvement

The multivariate LSTM incorporating pricing and promotional information achieved the lowest MAE among the tested models.

### Extreme demand spikes remain challenging

Although the final model captures the overall demand pattern, it tends to underestimate some extreme demand spikes. This is expected given the highly skewed nature of the target distribution.

---

# Final Model

Based on the evaluation results, **V3 — Multivariate LSTM** was selected as the final forecasting model.

### Final Model Specifications

| Parameter | Value |
|---|---|
| Model | Multivariate LSTM |
| Lookback | 8 weeks |
| LSTM Units | 64 |
| Dropout | 0.2 |
| Dense Units | 32 |
| Target Transformation | `log1p` |
| Optimizer | Adam |
| Loss | MSE |
| Evaluation Metric | MAE |
| Test MAE | **134.13** |
| Test RMSE | **335.70** |

The trained model is saved as:

```text
models/final_lstm_v3.keras
```

---

# Future Demand Forecasting

The final model was used to forecast demand for the future test period.

Future predictions cover:

```text
Weeks 146–155
```

The forecasting process follows a recursive strategy.

```text
Historical Demand
       ↓
Last 8 Weeks
       ↓
V3 LSTM
       ↓
Next Week Prediction
       ↓
Prediction Added to History
       ↓
Next 8-Week Sequence
       ↓
Next Week Prediction
       ↓
Repeat
```

This allows predictions to be generated sequentially across the future forecasting horizon.

The final forecast contains:

```text
32,573 predictions
```

and contains no missing prediction values.

The final output is saved as:

```text
outputs/final_submission.csv
```

---

# Streamlit Application

A Streamlit application is included to provide an interactive interface for the trained forecasting model.

The application allows users to:

- Select a fulfilment center
- Select a meal
- View historical demand
- Inspect recent demand-related features
- Generate a next-week demand prediction
- View final model information

Run the application using:

```bash
streamlit run app.py
```

---

# Project Structure

```text
Food Demand Forecasting/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── meal_info.csv
│   ├── fulfilment_center_info.csv
│   └── sample_submission.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_LSTM_Model.ipynb
│
├── models/
│   └── final_lstm_v3.keras
│
├── outputs/
│   └── final_submission.csv
│
├── cleaned_food_delivery.csv
├── app.py
├── requirements.txt
└── README.md
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras
- LSTM
- Jupyter Notebook
- Streamlit

---

# Skills Demonstrated

- Exploratory Data Analysis
- Data Cleaning
- Data Integration
- Time-Series Analysis
- Feature Engineering
- Target Transformation
- Feature Scaling
- Sequence Generation
- Deep Learning
- LSTM Forecasting
- Model Evaluation
- Baseline Modeling
- Recursive Forecasting
- Streamlit Application Development
- End-to-End Machine Learning Workflow

---

# Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Food-Demand-Forecasting
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Notebooks

The project contains two notebooks.

### EDA

```text
notebooks/01_EDA.ipynb
```

This notebook contains:

- Data inspection
- Data cleaning
- Dataset integration
- Demand analysis
- Center and meal analysis
- Price analysis
- Promotion analysis
- Time-series analysis
- ACF/PACF analysis

### LSTM Modeling

```text
notebooks/02_LSTM_Model.ipynb
```

This notebook contains:

- Time-series preparation
- Train/validation/test splitting
- Sequence generation
- Baseline forecasting
- V1 LSTM
- V2 LSTM
- V3 Multivariate LSTM
- Model evaluation
- Model comparison
- Future demand forecasting
- Submission generation

---

# Running the Streamlit App

After installing the dependencies:

```bash
streamlit run app.py
```

The application will open in the browser and provide an interactive demand forecasting interface.

---

# Future Improvements

Several improvements can be explored in future versions of the project:

- Incorporating meal category and cuisine as model features
- Adding city, region, and fulfilment-center type information
- Adding seasonal and calendar-based features
- Experimenting with longer lookback windows
- Hyperparameter optimization
- Testing GRU architectures
- Testing Bidirectional LSTM models
- Experimenting with attention mechanisms
- Comparing LSTM with XGBoost and other machine-learning forecasting approaches
- Developing center-level and meal-level demand dashboards
- Evaluating additional business-oriented forecasting metrics
- Improving prediction of extreme demand spikes

---

# Conclusion

This project presents an end-to-end approach to **food delivery demand forecasting using deep learning**.

The workflow combines exploratory data analysis, time-series analysis, baseline forecasting, target transformation, feature engineering, and LSTM-based sequence modeling.

Three LSTM approaches were evaluated, with the **multivariate LSTM achieving the lowest MAE of 134.13** among the tested models.

The final model uses an 8-week historical window together with demand, pricing, and promotional features to forecast future weekly meal demand.

The project demonstrates how historical operational data can be transformed into a forecasting system that can support **inventory planning, fulfilment-center operations, and demand-based decision-making**.
