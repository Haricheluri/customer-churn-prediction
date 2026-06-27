from flask import Blueprint, request, jsonify, render_template
from src.predict import predict_churn
from backend.recommendation import get_recommendation

main = Blueprint("main", __name__)


@main.route("/")
def home():
    return render_template("index.html")


@main.route("/health")
def health():
    return jsonify({
        "message": "Customer Churn Prediction API is Running"
    })


@main.route("/predict", methods=["POST"])
def predict():

    customer_data = request.form.to_dict()

    # Numerical features
    customer_data["monthly_spend"] = float(customer_data["monthly_spend"])
    customer_data["tenure_months_x"] = int(customer_data["tenure_months_x"])
    customer_data["num_products_owned"] = int(customer_data["num_products_owned"])
    customer_data["complaint_count"] = int(customer_data["complaint_count"])
    customer_data["satisfaction_score"] = int(customer_data["satisfaction_score"])
    customer_data["autopay"] = int(customer_data["autopay"])
    customer_data["age"] = int(customer_data["age"])

    # Predict
    result = predict_churn(customer_data)

    # Recommendation
    recommendation = get_recommendation(
        result["prediction"],
        result["probability"]
    )

    result.update(recommendation)

    return render_template(
        "result.html",
        result=result
    )