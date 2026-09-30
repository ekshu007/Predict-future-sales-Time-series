# Predict Future Sales — E-commerce Time-Series Forecasting

This repository contains an end-to-end machine learning pipeline to forecast monthly sales for **1C Company**, one of the largest software and entertainment retailers in Russia.

The project addresses common e-commerce forecasting challenges such as **extreme data sparsity, cold-start predictions for new inventory, and memory exhaustion** by engineering a heavily optimized Cartesian grid and utilizing **LightGBM** for global, multivariate time-series inference.

## 🚀 Project Architecture & Methodology

Traditional univariate time-series models such as ARIMA or Prophet can struggle at the granular, item-specific level of retail inventory due to data sparsity and scale. This project adopts a **tree-based machine learning approach** to capture cross-sectional catalog trends.

### 1. Sparsity Resolution — Cartesian Grid

The raw transaction log was converted into a **Cartesian product grid** of all active `shop_id` and `item_id` combinations per month.

This explicitly injected `0.0` target values into the dataset, eliminating survivorship bias and forcing the model to learn the specific conditions that correlate with zero demand.

### 2. Temporal Feature Engineering

Historical lag features were calculated across the approximately **11-million-row grid** to capture short-term momentum and annual seasonality.

The project uses:

* 1-month lag
* 2-month lag
* 3-month lag
* 6-month lag
* 12-month lag

These features allow the model to learn both recent sales behavior and longer-term seasonal patterns.

### 3. Overcoming the Cold-Start Problem

The test set requires predictions for newly released items that may have **zero historical sales**.

To provide fallback inference paths, text-based metadata was extracted from Russian labels and used to encode broader categorical information such as:

* `city_code`
* `main_type_code`
* `item_category_id`

This provides the model with information about an item even when its historical sales information is limited or unavailable.

### 4. Memory Optimization

Creating a large Cartesian grid can cause significant RAM consumption.

To prevent memory exhaustion, numerical columns were aggressively downcast using smaller data types such as:

* `int8`
* `int16`
* `float16`

This reduced the dataset's memory footprint by **more than 60%**.

---

## 💻 Tech Stack & Environment

| Category                | Technology                 |
| ----------------------- | -------------------------- |
| Language                | Python 3                   |
| Modeling                | LightGBM, Scikit-learn     |
| Data Manipulation       | Pandas, NumPy              |
| Development Environment | VS Code                    |
| Operating Environment   | Windows + WSL2             |
| Hardware                | NVIDIA RTX 3050 laptop GPU |

---

## 📊 Performance & Results

The global LightGBM regressor was trained using **leaf-wise growth** and **Gradient-based One-Side Sampling (GOSS)**.

### Validation Strategy

An expanding-window time-based split was used:

```text
Training:   Months 12–32
Validation: Month 33
Test:       Month 34
```

This prevents future information from leaking into the training data.

### Evaluation Metric

The primary evaluation metric is:

**Root Mean Squared Error (RMSE)**

Predictions were clipped to the range:

```text
0 ≤ prediction ≤ 20
```

### Best Validation Result

```text
Best Validation RMSE: 0.7182
Best Iteration: 211
```

Early stopping was used to prevent unnecessary training once validation performance stopped improving.

### Feature Importance

The model placed significant importance on:

* `item_cnt_month_lag_1`
* `item_category_id`
* Other historical sales features
* Engineered categorical groupings

The **1-month historical sales lag** was particularly influential in predicting future demand.

---

## 📂 Repository Structure

```text
├── data/
│   ├── raw/
│   │   └── # Original extracted Kaggle CSVs (git-ignored)
│   │
│   └── processed/
│       └── # Engineered data grids and submission files
│
├── notebooks/
│   └── 01_EDA.ipynb
│       # Outlier visualization and data distribution analysis
│
├── src/
│   ├── feature_engineering.ipynb
│   │   # Grid generation, lag features, and text extraction
│   │
│   ├── train.py
│   │   # Time-based splitting and LightGBM model training
│   │
│   └── predict.py
│       # Test set inference and Kaggle submission generation
│
├── .env
├── .gitignore
└── README.md
```

---

## ⚙️ How to Run

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Configure the Kaggle API

Ensure that your Kaggle API credentials are configured.

### 3. Download the Dataset

```bash
kaggle competitions download -c competitive-data-science-predict-future-sales
```

### 4. Build the Feature-Engineered Dataset

Open and execute:

```text
src/feature_engineering.ipynb
```

This generates the memory-optimized training grid:

```text
train_data.pkl
```

### 5. Train the LightGBM Model

```bash
python src/train.py
```

This trains the model and generates the feature-importance visualization.

### 6. Generate Predictions

```bash
python src/predict.py
```

The final predictions are saved as:

```text
submission.csv
```

---

## 🎯 Project Objective

The primary objective is to demonstrate how **machine learning can be applied to large-scale retail time-series forecasting** when traditional univariate forecasting approaches become difficult to scale.

The project combines:

* Time-series analysis
* Feature engineering
* Global machine learning models
* Categorical feature encoding
* Cold-start handling
* Memory optimization
* Time-aware validation
* Large-scale model training

This provides an end-to-end framework for predicting future sales across multiple stores and products.
