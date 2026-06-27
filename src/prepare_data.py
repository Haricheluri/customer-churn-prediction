"""
prepare_data.py
"""

import os
import pandas as pd

def prepare_data():

    print("Loading datasets...")

    churn = pd.read_csv("data/raw/customer_churn.csv")
    customers = pd.read_csv("data/raw/customers.csv")
    engagement = pd.read_csv("data/raw/customer_engagement_metrics.csv")
    rfm = pd.read_csv("data/raw/customer_rfm.csv")
    orders = pd.read_csv("data/raw/orders.csv")
    billing = pd.read_csv("data/raw/subscription_billing.csv")
    tickets = pd.read_csv("data/raw/support_tickets.csv")
    campaign = pd.read_csv("data/raw/campaign_responses.csv")

    os.makedirs("data/processed", exist_ok=True)

    customers["country"] = customers["country"].fillna(customers.groupby("state")["country"].transform(lambda x: x.mode()[0]))

    orders["shipping_country"] = orders["shipping_country"].fillna(orders.groupby("shipping_state")["shipping_country"].transform(lambda x: x.mode()[0]))

    orders["promo_code"] = orders["promo_code"].fillna("No Promo")

    customers.to_csv("data/processed/customers_cleaned.csv", index=False)

    orders.to_csv("data/processed/orders_cleaned.csv", index=False)

    print("Customers cleaned and saved.")
    print("Orders cleaned and saved.")

    print("Preparing integrated dataset...")

    churn_agg=churn.sort_values("updated_at").groupby("customer_id").last().reset_index()

    engagement_agg=engagement.sort_values("measured_date").groupby("customer_id").last().reset_index()

    rfm_agg=rfm.groupby("customer_id").agg({
        "recency_days":"mean",
        "frequency":"mean",
        "monetary":"mean",
        "r_score":"mean",
        "f_score":"mean",
        "m_score":"mean"
    }).reset_index()

    orders_agg=orders.groupby("customer_id").agg({
        "order_id":"count",
        "total_amount":"sum",
        "discount_amount":"sum",
        "shipping_cost":"sum",
        "is_gift":"sum"
    }).reset_index()

    orders_agg.rename(columns={
        "order_id":"total_orders",
        "total_amount":"total_revenue",
        "discount_amount":"total_discount",
        "shipping_cost":"total_shipping_cost",
        "is_gift":"gift_orders"
    },inplace=True)

    tickets_agg=tickets.groupby("customer_id").agg({
        "ticket_id":"count",
        "resolution_time_hours":"mean"
    }).reset_index()

    tickets_agg.rename(columns={
        "ticket_id":"total_tickets",
        "resolution_time_hours":"avg_resolution_time"
    },inplace=True)

    campaign_agg=campaign.groupby("customer_id").agg({
        "campaign_id":"count",
        "responded":"sum",
        "converted":"sum",
        "revenue":"sum"
    }).reset_index()

    campaign_agg.rename(columns={
        "campaign_id":"total_campaigns",
        "revenue":"campaign_revenue"
    },inplace=True)

    final_df=churn_agg.copy()

    merge_datasets={
        "customers":customers,
        "billing":billing,
        "engagement":engagement_agg,
        "rfm":rfm_agg,
        "orders":orders_agg,
        "tickets":tickets_agg,
        "campaign":campaign_agg
    }

    for name,df in merge_datasets.items():
        final_df=final_df.merge(
            df,
            on="customer_id",
            how="left"
        )

    num_cols=final_df.select_dtypes(include=["int64","float64"]).columns
    final_df[num_cols]=final_df[num_cols].fillna(0)

    cat_cols=final_df.select_dtypes(include="object").columns
    final_df[cat_cols]=final_df[cat_cols].fillna("Unknown")

    final_df.to_csv(
        "data/processed/final_customer_data.csv",
        index=False
    )

    print("Final dataset saved.")
    print("Shape:",final_df.shape)

prepare_data()
