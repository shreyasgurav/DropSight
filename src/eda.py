"""
eda.py
------
Generates Exploratory Data Analysis charts from the cleaned dataset
and saves them as PNG files in the /charts folder.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------- paths ---------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_CSV = os.path.join(BASE_DIR, "data", "cleaned_data.csv")
CHARTS_DIR = os.path.join(BASE_DIR, "output", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# --------------- style ---------------
sns.set_theme(style="whitegrid", palette="muted")
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974",
          "#64B5CD", "#E5AE38", "#6D904F", "#8B8B8B", "#810F7C"]


def load_data() -> pd.DataFrame:
    """Load the cleaned CSV."""
    return pd.read_csv(CLEAN_CSV)


# ======================= chart functions =======================

def chart_dropoff_per_page(df: pd.DataFrame) -> None:
    """Bar chart: Drop-off count per page (1–5)."""
    dropoff = df[df["dropped_off"] == 1].groupby("page").size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=dropoff, x="page", y="count", palette=COLORS, ax=ax)
    ax.set_title("Drop-Off Count per Page", fontsize=14, fontweight="bold")
    ax.set_xlabel("Page")
    ax.set_ylabel("Number of Drop-Offs")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "dropoff_per_page.png"), dpi=150)
    plt.close(fig)
    print("✅ dropoff_per_page.png saved")


def chart_category_distribution(df: pd.DataFrame) -> None:
    """Pie chart: Distribution of main categories browsed."""
    counts = df["main_category"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
           startangle=140, colors=COLORS[:len(counts)],
           textprops={"fontsize": 11})
    ax.set_title("Distribution of Main Categories", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "category_distribution.png"), dpi=150)
    plt.close(fig)
    print("✅ category_distribution.png saved")


def chart_sessions_by_country(df: pd.DataFrame) -> None:
    """Bar chart: Top 10 countries by number of sessions."""
    top = (df.groupby("country")["session_id"].nunique()
             .sort_values(ascending=False).head(10).reset_index())
    top.columns = ["country", "sessions"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top, x="country", y="sessions", palette=COLORS, ax=ax)
    ax.set_title("Top 10 Countries by Number of Sessions", fontsize=14, fontweight="bold")
    ax.set_xlabel("Country Code")
    ax.set_ylabel("Unique Sessions")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "sessions_by_country.png"), dpi=150)
    plt.close(fig)
    print("✅ sessions_by_country.png saved")


def chart_sessions_per_month(df: pd.DataFrame) -> None:
    """Line chart: Number of sessions per month."""
    monthly = (df.groupby("month")["session_id"].nunique()
                 .reset_index().rename(columns={"session_id": "sessions"}))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(monthly["month"], monthly["sessions"], marker="o",
            color=COLORS[0], linewidth=2, markersize=8)
    ax.set_title("Sessions per Month", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Unique Sessions")
    ax.set_xticks(monthly["month"])
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "sessions_per_month.png"), dpi=150)
    plt.close(fig)
    print("✅ sessions_per_month.png saved")


def chart_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap: Correlation matrix of all numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)
    print("✅ correlation_heatmap.png saved")


def chart_dropoff_by_category(df: pd.DataFrame) -> None:
    """Bar chart: Drop-off rate by main category."""
    rate = (df.groupby("main_category")["dropped_off"].mean()
              .sort_values(ascending=False).reset_index())
    rate.columns = ["main_category", "dropoff_rate"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=rate, x="main_category", y="dropoff_rate", palette=COLORS, ax=ax)
    ax.set_title("Drop-Off Rate by Main Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Main Category")
    ax.set_ylabel("Drop-Off Rate")
    ax.set_ylim(0, 1)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "dropoff_by_category.png"), dpi=150)
    plt.close(fig)
    print("✅ dropoff_by_category.png saved")


def chart_avg_order_steps_by_country(df: pd.DataFrame) -> None:
    """Bar chart: Average order steps before drop-off by country (top 10)."""
    drop_df = df[df["dropped_off"] == 1]
    avg = (drop_df.groupby("country")["order"].mean()
                  .sort_values(ascending=False).head(10).reset_index())
    avg.columns = ["country", "avg_order_steps"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=avg, x="country", y="avg_order_steps", palette=COLORS, ax=ax)
    ax.set_title("Avg Order Steps Before Drop-Off (Top 10 Countries)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Country Code")
    ax.set_ylabel("Avg Order Steps")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "avg_order_steps_by_country.png"), dpi=150)
    plt.close(fig)
    print("✅ avg_order_steps_by_country.png saved")


def chart_dropoff_by_price(df: pd.DataFrame) -> None:
    """Bar chart: Drop-off rate by price range (Low vs High)."""
    rate = (df.groupby("price_range")["dropped_off"].mean()
              .reset_index().rename(columns={"dropped_off": "dropoff_rate"}))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=rate, x="price_range", y="dropoff_rate", palette=COLORS[:2], ax=ax)
    ax.set_title("Drop-Off Rate by Price Range", fontsize=14, fontweight="bold")
    ax.set_xlabel("Price Range")
    ax.set_ylabel("Drop-Off Rate")
    ax.set_ylim(0, 1)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "dropoff_by_price.png"), dpi=150)
    plt.close(fig)
    print("✅ dropoff_by_price.png saved")


# ======================= run all =======================

def generate_all_charts() -> None:
    """Generate every EDA chart."""
    df = load_data()
    chart_dropoff_per_page(df)
    chart_category_distribution(df)
    chart_sessions_by_country(df)
    chart_sessions_per_month(df)
    chart_correlation_heatmap(df)
    chart_dropoff_by_category(df)
    chart_avg_order_steps_by_country(df)
    chart_dropoff_by_price(df)
    print("\n🎉 All EDA charts generated!")


if __name__ == "__main__":
    generate_all_charts()
