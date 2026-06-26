# Customer Churn Prediction & Retention Recommendation System

A Machine Learning web application that predicts whether a customer is likely to churn and provides personalized retention recommendations based on the predicted risk level.

---

## Project Overview

Customer churn is one of the biggest challenges for subscription-based and service-oriented businesses. This project predicts customer churn using Machine Learning and recommends suitable retention strategies to reduce customer loss.

---

## Features

- Customer churn prediction
- Churn probability estimation
- Risk level classification
- Personalized retention recommendations
- User-friendly Flask web application
- MLflow experiment tracking
- LightGBM machine learning model

---

## Tech Stack

### Machine Learning

- Python
- Scikit-learn
- LightGBM
- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Backend

- Flask

### Frontend

- HTML
- CSS

### Experiment Tracking

- MLflow

---

## Project Structure

```
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
├── models/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb
│
├── src/
│   └── predict.py
│
├── data/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Machine Learning Workflow

- Data Collection
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
- Feature Selection
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- MLflow Experiment Tracking
- Model Deployment using Flask

---

## Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | 92.82% |
| Precision | 100% |
| Recall | 83.37% |
| F1 Score | 90.93% |
| ROC-AUC | 91.57% |
| Cross Validation Accuracy | 92.72% |

---

## Risk Categories

| Probability | Risk Level |
|-------------|------------|
| 0–20% | Very Low |
| 20–40% | Low |
| 40–60% | Medium |
| 60–80% | High |
| 80–100% | Very High |

---

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
```

Navigate to the project

```bash
cd customer-churn-prediction
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Flask application

```bash
python -m backend.app
```

Open

```
http://127.0.0.1:5000
```

---

## MLflow

Start the MLflow UI

```bash
mlflow ui
```

Open

```
http://127.0.0.1:5000
```

---

## Future Enhancements

- Customer segmentation
- SHAP explainability
- Docker support
- Azure App Service deployment
- CI/CD using Azure DevOps
- Real-time prediction API

---

## Author

**Hari Haran**

B.Tech – Computer Science & Data Science

GitHub: https://github.com/Haricheluri

---

## License

This project is developed for educational and portfolio purposes.