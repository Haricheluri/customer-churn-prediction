import joblib
import pandas as pd

# Load trained pipeline
pipeline = joblib.load("models/churn_pipeline.pkl")

# Template containing all feature columns
template = pd.read_csv("models/template_customer.csv")


def predict_churn(customer_data):
    """
    Predict customer churn probability.
    """

    df = template.copy()

    for key, value in customer_data.items():
        if key in df.columns:
            df.loc[0, key] = value

    prediction = pipeline.predict(df)[0]

    probability = pipeline.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4)
    }