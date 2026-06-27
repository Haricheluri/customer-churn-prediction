# Customer Churn Prediction

## Overview

Customer Churn Prediction is an end-to-end Machine Learning project that predicts whether a customer is likely to churn based on customer demographics, subscription details, billing information, engagement metrics, purchasing behavior, and campaign responses.

The project includes:

* Data Preparation Pipeline
* Data Preprocessing
* Feature Engineering
* Model Training & Comparison
* MLflow Experiment Tracking
* Flask Web Application
* Customer Retention Recommendations

---

# Project Structure

```text
customer-churn-prediction/
│
├── backend/
│   ├── app.py
│   ├── routes.py
│   └── recommendation.py
│
├── frontend/
│   ├── index.html
│   ├── result.html
│   └── style.css
│
├── src/
│   ├── prepare_data.py
│   ├── data_preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── churn_pipeline.pkl
│   └── template_customer.csv
│
├── notebooks/
├── mlruns/
├── requirements.txt
└── README.md
```

---

# Workflow

1. Load raw datasets
2. Clean customer and order data
3. Merge all datasets
4. Perform feature engineering
5. Preprocess data
6. Compare multiple ML models
7. Select the best-performing model
8. Save trained pipeline
9. Deploy with Flask
10. Predict customer churn

---

# Models Compared

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM

The model with the highest cross-validation accuracy is automatically selected and saved.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* XGBoost
* MLflow
* Flask
* HTML
* CSS

---

# How to Run

## 1. Clone Repository

```bash
git clone <repository-url>
cd customer-churn-prediction
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Prepare Dataset

```bash
python -m src.prepare_data
```

This command:

* Cleans datasets
* Creates processed datasets
* Generates `final_customer_data.csv`

---

## 5. Train Model

```bash
python -m src.train
```

This command:

* Performs preprocessing
* Compares multiple ML models
* Tracks experiments using MLflow
* Saves the trained pipeline
* Creates prediction template

Generated files:

```
models/churn_pipeline.pkl

models/template_customer.csv
```

---

## 6. Start Flask Application

```bash
python -m backend.app
```

Open:

```
http://127.0.0.1:5000
```

---

## 7. View MLflow Dashboard

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

# Prediction Inputs

The web application predicts customer churn using:

* Monthly Spend
* Tenure
* Contract Type
* Payment Method
* Number of Products
* Complaint Count
* Satisfaction Score
* Autopay
* Age
* Gender
* Customer Segment

The remaining model features are automatically populated using a template dataset.

---

# Project Outputs

* Customer Churn Prediction
* Churn Probability
* Risk Level
* Retention Recommendation

---

# About the Notebooks

The Jupyter notebooks included in this project are provided for experimentation, exploratory data analysis (EDA), feature engineering, and model development.

The production pipeline does **not** depend on the notebooks.

All required functionality has been converted into reusable Python scripts located in the `src/` directory.

To reproduce the project, users only need to execute:

```bash
python -m src.prepare_data
python -m src.train
python -m backend.app
```

without running any notebook.

---

# Notes

* All preprocessing steps are included inside the training pipeline.
* The saved pipeline performs preprocessing and prediction together.
* Unknown categorical values are handled automatically.
* MLflow is used for experiment tracking.
* Random Forest is automatically selected if it achieves the best validation performance.

---

# Author

Hari Haran Cheluru
