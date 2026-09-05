# Return Guard — Explainable Return-Risk Scoring with Financial Impact

**One class of loss, one working model, real held-out metrics.** Return Guard predicts which orders are likely to become costly returns, explains every score with SHAP, and hands the decision to a human merchant instead of auto-blocking anything.

Built for the "AI Risk Manager" track: *stop the merchant losing money to fraud, returns and chargebacks — build a working detector, verifier or auto-responder for one class of loss, with measured precision and recall on a held-out test set.*

**[Open the live dashboard](./dashboard/index.html)** (or see [screenshots](#dashboard) below).

---

## Why this scope

The track asks for **one** class of loss with **honest, measured** metrics. Return risk is a clean tabular classification problem — it's the loss class where precision/recall/false-positive-cost can be reported without hand-waving. So this repo goes deep on one engine instead of wide across four. See [Roadmap](#roadmap--the-other-three-engines) for how this fits a larger platform.

## What it does

1. **Predicts** return probability per order using gradient-boosted trees (XGBoost) trained on customer history, product category, order value, discount, and payment signals.
2. **Explains** every prediction with SHAP — the merchant sees *why* an order is risky, not just a number.
3. **Quantifies** the financial trade-off at any threshold: loss prevented vs. false-positive friction cost vs. manual review cost vs. loss still missed. This is a live slider in the dashboard, not a static slide.
4. **Recommends, never auto-blocks.** Flagged orders go to a review queue where a human picks Approve / Ask for verification / Hold — every decision is logged.
5. **Reports itself honestly.** A Model Health screen shows ROC-AUC, precision/recall/F1, the confusion matrix, and the train/test split — computed on the 20% of data the model never saw during training.

## Repository structure

```
return-guard/
├── data/
│   └── orders.csv              # generated synthetic dataset (see below)
├── ml/
│   ├── generate_data.py        # synthetic e-commerce returns dataset generator
│   └── train_model.py          # XGBoost training, held-out eval, SHAP, exports dashboard data
├── dashboard/
│   ├── index.html              # dashboard shell
│   ├── style.css
│   ├── app.js                  # slider simulation, risk queue, charts
│   └── data.js                 # generated — model outputs consumed by the dashboard
├── requirements.txt
└── README.md
```

## About the dataset

This uses a **synthetic** dataset (`ml/generate_data.py`), not real merchant data — stated plainly, not hidden. It's generated with hand-tuned, realistic correlations (category return rates, customer return history, discount depth, payment method) so the model has genuine signal to learn and the metrics reflect real model behavior, not a rigged demo. Swapping in a real merchant's order history means pointing `train_model.py` at a CSV with the same column names — no other code changes needed.

The resulting dataset: ~9,000 orders, ~20% overall return rate, category return rates ranging from ~13% (Beauty) to ~32% (Women's Apparel), matching the shape of real e-commerce return data.

## Running it yourself

```bash
pip install -r requirements.txt

cd ml
python generate_data.py     # writes ../data/orders.csv
python train_model.py       # trains, evaluates, writes ../dashboard/data.js
```

Then open `dashboard/index.html` directly in a browser — it's a static page, no server or build step required. (To publish it, push this repo and turn on GitHub Pages, serving from `/dashboard`.)

## The methodology, made explicit

- **80/20 train/test split, stratified, seeded, locked before any feature engineering.** The test set is touched exactly once, for final scoring.
- **The "default" operating threshold** shown in Model Health is the one maximizing F1 on the test set — reported as what it is (a post-hoc summary threshold), while the dashboard's actual interactive simulator sweeps *every* threshold from 5% to 95%, so nothing is cherry-picked for the headline number.
- **Financial assumptions are declared, not fitted:** return loss = 35% of order value, false-positive friction cost = 2% of order value, flat manual review cost = ₹25/order. These are constants at the top of `train_model.py` — change them to match a real merchant's numbers.
- **The LLM-adjacent SHAP explanations never invent anything.** Each factor shown is a real SHAP value for that exact prediction, not a templated sentence.

## Dashboard

Three views, one static page:

- **Impact Simulator** — drag the threshold, watch precision/recall/F1 and the four-part financial breakdown (prevented / false-positive cost / review cost / still missed) update live, computed from the real held-out sweep.
- **Risk Queue** — orders ranked by risk score, filterable by band, each expandable to show its top SHAP drivers and a human decision (Approve / Verify / Hold) that gets logged.
- **Model Health** — ROC-AUC, confusion matrix, train/test split, and top model drivers overall.

## Escalation tiers

| Band | Score | Action |
|---|---|---|
| 🟢 LOW | 0–29 | Background monitoring, no intervention |
| 🟡 MEDIUM | 30–59 | Enhanced monitoring, optional verification |
| 🟠 HIGH | 60–84 | Human review queue, evidence prepared |
| 🔴 CRITICAL | 85–100 | Priority human review |

## Roadmap — the other three engines

Return Guard is Engine 1 of a four-engine risk platform concept. The other three follow the same **detect → explain → quantify → human-decide** pattern, deliberately left out of this build so every number in this repo is real and defensible:

- **Fraud-spike detector** — rolling-baseline anomaly detection on transaction velocity/failure rate, flagging abnormal periods rather than individual transactions.
- **Abuse-ring sentinel** — graph analysis over accounts/devices/IPs/payment methods to surface coordinated account clusters.
- **Chargeback evidence responder** — retrieval, not classification: assembles verified transaction/order/delivery records into an evidence dossier, explicitly never inventing evidence the database doesn't contain.

Each would report its own honest metrics (precision/recall for the return/fraud engines, evidence-completeness and response-accuracy for the chargeback engine) rather than being folded into one composite score.

## Strictly defense-only

Return Guard never takes an autonomous action against a customer or transaction. It ranks, explains, and quantifies — every action is a human decision, logged in the review queue.
