"""
recommendations.py
-------------------
Business recommendation engine based on predicted drop-off page
and price range context.
"""


# Page-level recommendations
PAGE_RECOMMENDATIONS = {
    1: "Improve homepage layout and add trending products section",
    2: "Add better filters and category navigation",
    3: "Show discount popup or limited stock alert on product page",
    4: "Offer free shipping or show product reviews to build trust",
    5: "Send cart abandonment email or show exit-intent popup",
}

# Price-aware recommendation
HIGH_PRICE_TIP = "Suggest similar lower-priced alternatives"


def get_recommendation(predicted_page: int, price_range: str = "Low") -> dict:
    """
    Return a business recommendation based on the predicted drop-off page
    and the price range of the product being viewed.

    Parameters
    ----------
    predicted_page : int  (1–5)
    price_range    : str  ("Low" or "High")

    Returns
    -------
    dict with keys: page_recommendation, price_tip, full_recommendation
    """
    page_rec = PAGE_RECOMMENDATIONS.get(
        predicted_page,
        "No specific recommendation available for this page."
    )

    price_tip = ""
    if price_range == "High":
        price_tip = HIGH_PRICE_TIP

    full = page_rec
    if price_tip:
        full = f"{page_rec}. Additionally: {price_tip}."

    return {
        "page_recommendation": page_rec,
        "price_tip": price_tip,
        "full_recommendation": full,
    }


# --------------- quick test ---------------
if __name__ == "__main__":
    for pg in range(1, 6):
        for pr in ["Low", "High"]:
            rec = get_recommendation(pg, pr)
            print(f"Page {pg} | Price {pr:4s} → {rec['full_recommendation']}")
        print()
