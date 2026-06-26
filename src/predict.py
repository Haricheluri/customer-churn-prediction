import joblib
import pandas as pd

model = joblib.load("models/churn_model.pkl")

template = pd.read_csv("data/processed/template_customer.csv")


def predict_churn(customer_data):

    df = template.copy()

    for key, value in customer_data.items():
        df.loc[0, key] = value

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4)
    }