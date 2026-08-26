from pathlib import Path
import pickle

import pandas as pd
from scipy.sparse import csr_matrix
import implicit


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ORDERS_PATH = RAW_DIR / "orders.csv"
ORDER_PRODUCTS_PATH = RAW_DIR / "order_products__prior.csv"
PRODUCTS_PATH = RAW_DIR / "products.csv"

TRANSACTIONS_PATH = PROCESSED_DIR / "transactions_sampled.csv"
OUTPUT_PATH = PROJECT_ROOT / "app" / "als_artifacts.pkl"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Validate raw files
# ---------------------------------------------------------

required_files = [
    ORDERS_PATH,
    ORDER_PRODUCTS_PATH,
    PRODUCTS_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


# ---------------------------------------------------------
# Build or load sampled transactions
# ---------------------------------------------------------

if TRANSACTIONS_PATH.exists():

    print("Loading existing sampled transactions...")

    transactions = pd.read_csv(
        TRANSACTIONS_PATH,
        usecols=["user_id", "product_id"],
    )

else:

    print("Creating a reproducible sample of 200,000 prior orders...")

    orders = pd.read_csv(
        ORDERS_PATH,
        usecols=["order_id", "user_id", "eval_set"],
    )

    prior_orders = orders[orders["eval_set"] == "prior"].copy()

    sampled_order_ids = (
        pd.Series(prior_orders["order_id"].unique())
        .sample(
            n=min(200_000, prior_orders["order_id"].nunique()),
            random_state=42,
        )
    )

    sampled_order_set = set(sampled_order_ids)

    prior_orders = prior_orders[
        prior_orders["order_id"].isin(sampled_order_set)
    ][["order_id", "user_id"]]

    print("Reading order-product data in chunks...")

    filtered_chunks = []

    for chunk in pd.read_csv(
        ORDER_PRODUCTS_PATH,
        usecols=["order_id", "product_id"],
        chunksize=1_000_000,
    ):

        filtered = chunk[
            chunk["order_id"].isin(sampled_order_set)
        ]

        if not filtered.empty:
            filtered_chunks.append(filtered)

    order_products_sample = pd.concat(
        filtered_chunks,
        ignore_index=True,
    )

    transactions = order_products_sample.merge(
        prior_orders,
        on="order_id",
        how="inner",
    )

    transactions.to_csv(
        TRANSACTIONS_PATH,
        index=False,
    )

    print(
        f"Sampled transactions saved to:\n{TRANSACTIONS_PATH}"
    )


# ---------------------------------------------------------
# Product lookup
# ---------------------------------------------------------

products = pd.read_csv(
    PRODUCTS_PATH,
    usecols=["product_id", "product_name"],
)

product_lookup = (
    products
    .set_index("product_id")["product_name"]
    .to_dict()
)


# ---------------------------------------------------------
# Build user-item interaction matrix
# ---------------------------------------------------------

print("Building user-item matrix...")

interaction_counts = (
    transactions
    .groupby(["user_id", "product_id"])
    .size()
    .reset_index(name="interaction")
)

user_ids = interaction_counts["user_id"].unique()
product_ids = interaction_counts["product_id"].unique()

user_to_idx = {
    user_id: idx
    for idx, user_id in enumerate(user_ids)
}

product_to_idx = {
    product_id: idx
    for idx, product_id in enumerate(product_ids)
}

idx_to_product = {
    idx: product_id
    for product_id, idx in product_to_idx.items()
}

interaction_counts["user_idx"] = (
    interaction_counts["user_id"].map(user_to_idx)
)

interaction_counts["product_idx"] = (
    interaction_counts["product_id"].map(product_to_idx)
)

user_item_matrix = csr_matrix(
    (
        interaction_counts["interaction"].astype(float),
        (
            interaction_counts["user_idx"],
            interaction_counts["product_idx"],
        ),
    ),
    shape=(len(user_ids), len(product_ids)),
)

print("User-item matrix shape:", user_item_matrix.shape)


# ---------------------------------------------------------
# Train ALS model
# ---------------------------------------------------------

print("Training ALS model...")

als_model = implicit.als.AlternatingLeastSquares(
    factors=50,
    regularization=0.01,
    iterations=20,
    random_state=42,
)

als_model.fit(user_item_matrix)


# ---------------------------------------------------------
# Save Streamlit artifacts
# ---------------------------------------------------------

artifacts = {
    "als_model": als_model,
    "user_to_idx": user_to_idx,
    "idx_to_product": idx_to_product,
    "user_item_matrix": user_item_matrix,
    "product_lookup": product_lookup,
}

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(artifacts, f)

print("Corrected ALS artifact saved successfully.")
print(OUTPUT_PATH)