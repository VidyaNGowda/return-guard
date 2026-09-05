"""
Return Guard — synthetic dataset generator.

Generates a realistic e-commerce order dataset with a binary `returned` label.
Feature/label correlations are hand-tuned to mimic well-known return-risk
drivers (category return rates, customer return history, order value,
discount usage, size-sensitive categories) so a tree model has real signal
to learn, without using any real merchant data.

Run: python generate_data.py
Output: ../data/orders.csv
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_ORDERS = 9000

CATEGORIES = {
    # category: (base_return_rate, avg_price)
    "Apparel - Womens":   (0.28, 1800),
    "Apparel - Mens":     (0.19, 1600),
    "Footwear":           (0.24, 2400),
    "Electronics":        (0.09, 6500),
    "Home & Kitchen":     (0.07, 2200),
    "Beauty":             (0.05, 900),
    "Mobile Accessories": (0.06, 700),
    "Furniture":          (0.11, 9500),
}
CATEGORY_NAMES = list(CATEGORIES.keys())

PAYMENT_METHODS = ["COD", "UPI", "Credit Card", "Debit Card", "Netbanking"]
PAYMENT_RETURN_BUMP = {"COD": 0.09, "UPI": 0.0, "Credit Card": -0.01, "Debit Card": 0.0, "Netbanking": -0.02}

n_customers = 2200
customer_ids = np.arange(1, n_customers + 1)
# each customer has a latent "return propensity" (most are low, a tail is high)
customer_propensity = np.clip(RNG.beta(1.6, 7.0, n_customers), 0, 0.9)
customer_order_count_hist = RNG.poisson(9, n_customers) + 3
customer_prior_returns = RNG.binomial(customer_order_count_hist, customer_propensity)

rows = []
for i in range(N_ORDERS):
    cust_idx = RNG.integers(0, n_customers)
    cust_id = customer_ids[cust_idx]
    propensity = customer_propensity[cust_idx]
    prior_orders = int(customer_order_count_hist[cust_idx])
    prior_returns = int(customer_prior_returns[cust_idx])
    prior_return_ratio = prior_returns / max(prior_orders, 1)

    category = RNG.choice(CATEGORY_NAMES)
    base_rate, avg_price = CATEGORIES[category]

    order_value = float(np.clip(RNG.normal(avg_price, avg_price * 0.4), 150, avg_price * 4))
    discount_pct = float(np.clip(RNG.beta(1.5, 6) * 60, 0, 60))  # most orders small discount, some big
    payment_method = RNG.choice(PAYMENT_METHODS, p=[0.22, 0.34, 0.24, 0.14, 0.06])
    days_since_last_order = int(np.clip(RNG.exponential(35), 0, 400))
    order_freq_30d = int(RNG.poisson(1.2))  # orders by this customer in last 30 days

    # ---- latent return probability model (ground truth generating process) ----
    logit = (
        -3.3
        + 5.6 * base_rate
        + 3.4 * prior_return_ratio
        + 0.5 * propensity
        + PAYMENT_RETURN_BUMP[payment_method]
        + 0.00009 * (order_value - avg_price)
        + 0.014 * discount_pct
        + 0.12 * min(order_freq_30d, 5)
        - 0.002 * min(days_since_last_order, 120)
    )
    prob = 1 / (1 + np.exp(-logit))
    returned = int(RNG.random() < prob)

    rows.append(dict(
        order_id=100000 + i,
        customer_id=int(cust_id),
        category=category,
        order_value=round(order_value, 2),
        discount_pct=round(discount_pct, 1),
        payment_method=payment_method,
        prior_orders=prior_orders,
        prior_returns=prior_returns,
        prior_return_ratio=round(prior_return_ratio, 3),
        category_return_rate=base_rate,
        days_since_last_order=days_since_last_order,
        order_freq_30d=order_freq_30d,
        returned=returned,
    ))

df = pd.DataFrame(rows)
df.to_csv("../data/orders.csv", index=False)
print(f"Wrote {len(df)} orders to ../data/orders.csv")
print(f"Overall return rate: {df['returned'].mean():.3f}")
print(df.groupby('category')['returned'].mean().sort_values(ascending=False))
