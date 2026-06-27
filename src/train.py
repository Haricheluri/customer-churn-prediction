"""
train.py

Production training script for Customer Churn Prediction.
"""

import os
import warnings
import joblib
import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.data_preprocessing import load_and_preprocess_data

warnings.filterwarnings("ignore")


def train():

    # Load data and preprocessing pipeline
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()

    # Feature Selection
    selector = SelectKBest(
        score_func=f_classif,
        k=50,
    )

    # Models to compare
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),        
        "Random Forest": RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=42,
        ),

        "LightGBM": LGBMClassifier(
            random_state=42,
            verbose=-1,
        ),
    }

    # Hyperparameter grids
    param_grids = {

        "Random Forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 20],
            "model__min_samples_split": [2, 5],
        },

        "XGBoost": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [4, 6],
        },

        "LightGBM": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [5, 10],
        },
    }

    # -----------------------------
    # Model Comparison
    # -----------------------------

    best_pipeline = None
    best_model_name = None
    best_score = float("-inf")

    print("\nComparing Models...\n")

    for name, model in models.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("selector", selector),
                ("model", model),
            ]
        )

        # Logistic Regression (No GridSearch)
        if name == "Logistic Regression":

            pipeline.fit(X_train, y_train)

            score = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=5,
                scoring="accuracy",
                n_jobs=-1,
            ).mean()

            fitted_pipeline = pipeline

        # Tree Models (GridSearchCV)
        else:

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grids[name],
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
            )

            grid.fit(X_train, y_train)

            score = grid.best_score_
            fitted_pipeline = grid.best_estimator_

        print(f"{name:<22}: {score:.4f}")

        if score > best_score:

            best_score = score
            best_pipeline = fitted_pipeline
            best_model_name = name

    print("\n" + "=" * 50)
    print("Model Comparison Results")
    print("=" * 50)
    print(f"Best Model      : {best_model_name}")
    print(f"Best CV Accuracy: {best_score:.4f}")
    print("=" * 50)

    # -----------------------------
    # Evaluate Best Model
    # -----------------------------

    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    # -----------------------------
    # Evaluation Metrics
    # -----------------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    # -----------------------------
    # Confusion Matrix
    # -----------------------------

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

    # -----------------------------
    # MLflow
    # -----------------------------

    tracking_dir = os.path.abspath("mlruns")

    mlflow.set_tracking_uri(
        f"file:///{tracking_dir.replace(os.sep,'/')}"
    )

    mlflow.set_experiment(
        "Customer Churn Prediction"
    )

    with mlflow.start_run(run_name=best_model_name):

        mlflow.log_param("best_model",best_model_name)
        mlflow.log_metric("cv_accuracy",best_score)
        mlflow.log_metric( "accuracy",accuracy)
        mlflow.log_metric("precision",precision)
        mlflow.log_metric( "recall", recall)
        mlflow.log_metric("f1_score",f1)
        mlflow.log_metric("roc_auc",roc_auc)
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.sklearn.log_model(best_pipeline,"model")

    # -----------------------------
    # Save Pipeline
    # -----------------------------

    os.makedirs("models",exist_ok=True,)

    joblib.dump(
        best_pipeline,
        "models/churn_pipeline.pkl",
    )

    # -----------------------------
    # Save Template for Prediction
    # -----------------------------

    X_train.iloc[[0]].to_csv(
        "models/template_customer.csv",
        index=False,
    )

    # -----------------------------
    # Final Output
    # -----------------------------

    print("\nTraining Completed Successfully")
    print("-" * 45)
    print(f"Best Model : {best_model_name}")
    print(f"CV Accuracy: {best_score:.4f}")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print(f"ROC AUC    : {roc_auc:.4f}")
    print("-" * 45)

    print("Pipeline Saved : models/churn_pipeline.pkl")
    print("Template Saved : models/template_customer.csv")
    print("MLflow Logs    : mlruns/")


if __name__ == "__main__":
    train()