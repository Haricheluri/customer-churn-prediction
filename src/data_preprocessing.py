import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_and_preprocess_data(data_path="data/processed/final_customer_data.csv"):
    """
    Loads customer data, performs feature engineering,
    and returns train/test data along with the preprocessing pipeline.
    """

    # Load dataset
    df = pd.read_csv(data_path)

    # -----------------------------
    # Remove unwanted columns
    # -----------------------------

    remove_cols = [

        # Target leakage
        "predicted_churn_prob",
        "churn_reason",

        # Highly correlated with target
        "days_since_last_order",
        "support_tickets_30d",
        "nps_score_x",

        # IDs
        "customer_id",
        "billing_id",
        "metric_id",

        # Personal information
        "name",
        "first_name",
        "last_name",
        "email",
        "phone",
        "street_address",
        "postal_code",

        # High-cardinality columns
        "occupation",
        "city",
        "state",
        "billing_address_city",
        "account_manager_x",
        "account_manager_y",

        # Duplicate columns
        "payment_method_y",
        "contract_type_y",
        "tenure_months_y",
        "nps_score_y",
    ]

    df.drop(columns=remove_cols, inplace=True)

    # -----------------------------
    # Date Feature Engineering
    # -----------------------------

    date_cols = [
        "signup_date",
        "last_contact_date",
        "contract_end_date",
        "measured_date",
        "created_at_x",
        "updated_at_x",
        "created_at_y",
        "updated_at_y",
        "last_login_date_x",
        "last_login_date_y",
    ]

    for col in date_cols:

        df[col] = pd.to_datetime(df[col], errors="coerce")

        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month

    df.drop(columns=date_cols, inplace=True)

    # -----------------------------
    # Features & Target
    # -----------------------------

    X = df.drop("churned", axis=1)
    y = df["churned"]

    # -----------------------------
    # Train Test Split
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # -----------------------------
    # Numerical & Categorical Columns
    # -----------------------------

    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    # -----------------------------
    # Numerical Pipeline
    # -----------------------------

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # -----------------------------
    # Categorical Pipeline
    # -----------------------------

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    # -----------------------------
    # Column Transformer
    # -----------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )


if __name__ == "__main__":

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = load_and_preprocess_data()

    print("Training Samples :", X_train.shape)
    print("Testing Samples  :", X_test.shape)