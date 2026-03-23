"""
data_loader.py
--------------
Loads the raw e-shop CSV, cleans column names, maps encoded values
to human-readable labels, engineers the 'dropped_off' feature,
and saves the result as cleaned_data.csv.
"""

import pandas as pd
import os

# --------------- paths ---------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(BASE_DIR, "data", "e-shop_clothing_2008.csv")
CLEAN_CSV = os.path.join(BASE_DIR, "data", "cleaned_data.csv")


def load_and_clean_data() -> pd.DataFrame:
    """Load raw CSV, rename columns, map values, engineer features."""

    # --- Load ---
    df = pd.read_csv(RAW_CSV, sep=";")

    # --- Rename columns ---
    df.rename(columns={
        "page 1 (main category)": "main_category",
        "page 2 (clothing model)": "clothing_model",
        "session ID": "session_id",
        "price 2": "price_range",
        "model photography": "photo_type",
    }, inplace=True)

    # --- Map encoded values to readable labels ---
    category_map = {1: "Trousers", 2: "Skirts", 3: "Blouses", 4: "Sale"}
    price_range_map = {1: "Low", 2: "High"}
    photo_type_map = {1: "En-face", 2: "Profile"}

    df["main_category"] = df["main_category"].map(category_map)
    df["price_range"] = df["price_range"].map(price_range_map)
    df["photo_type"] = df["photo_type"].map(photo_type_map)

    # --- Feature engineering: dropped_off ---
    # A row is considered a drop-off if its page equals the max page
    # reached in that session.
    max_page_per_session = df.groupby("session_id")["page"].transform("max")
    df["dropped_off"] = (df["page"] == max_page_per_session).astype(int)

    return df


def save_cleaned_data(df: pd.DataFrame) -> None:
    """Persist the cleaned DataFrame to CSV."""
    df.to_csv(CLEAN_CSV, index=False)
    print(f"✅ Cleaned data saved → {CLEAN_CSV}  ({len(df)} rows)")


# --------------- main ---------------
if __name__ == "__main__":
    df = load_and_clean_data()
    save_cleaned_data(df)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
