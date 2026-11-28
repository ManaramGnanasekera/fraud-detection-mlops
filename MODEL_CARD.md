# Model card: card fraud detector

| | |
|---|---|
| **Model** | `workspace.fraud.fraud_detector`, version 1, alias `champion` (Unity Catalog model registry) |
| **Type** | XGBoost binary classifier (400 trees, depth 6), with a cost-based alert threshold of **0.19** |
| **Owner** | Manaram Gnanasekera |
| **Status** | Portfolio project, trained on synthetic data. Not for use on real customers. |
| **Deployed as** | Serverless Amazon SageMaker endpoint `fraud-detector` (XGBoost 3.0-5 container, `deploy/inference.py`) |

## Intended use

Scores a card transaction at the moment it happens and returns a fraud probability and an alert flag. Flagged transactions go to a human fraud analyst, supported by an AI assistant that explains the score. The model **recommends**; a person **decides**. It is not designed to block transactions automatically, to make credit decisions, or to score anything other than card transactions.

## Training data

- [Kaggle Credit Card Transactions Fraud Detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection): **synthetic**, 1,852,394 transactions, 9,651 frauds (0.52%), January 2019 to December 2020.
- Customer names and street addresses are removed before modelling. Gender is kept only to measure fairness and is not a model input.
- Split by time: training Jan 2019 – Mar 2020, validation Apr – Jun 2020 (threshold choice), test Jul – Dec 2020 (final evaluation, used once).

## Inputs

26 inputs prepared from point-in-time behavioural features: transaction amount, the card's 30-day average spend and the ratio to it, transaction counts in the previous hour and 24 hours, time since the card's previous transaction, hour of day and a night flag, customer age, distance from home, city population, whether the card has 30-day history, and the spending category. Every window uses only earlier transactions, so no future information reaches the model.

## Performance (test months, Jul – Dec 2020)

| Approach | Recall | Precision | Alerts | Cost |
|---|---|---|---|---|
| Rules baseline | 52.0% | 13.7% | 7,639 | $322,627 |
| Logistic regression | 80.7% | 11.0% | 14,706 | $111,257 |
| **This model** | **95.9%** | **43.3%** | **4,459** | **$24,161** |

PR-AUC 0.93. Cost assumes a missed fraud loses its full amount and each false alarm costs $5. With no model, the loss would be $1,060,050.

## What drives the score

By mean absolute SHAP value: transaction amount, the card's usual spend, hour of day, amount relative to usual spend, and night-time. These match the signals found in exploratory analysis. The endpoint returns the top contributing inputs with every score, so each alert can be explained.

## Fairness

| Group | Recall | False alarm rate |
|---|---|---|
| Women | 93.9% | 0.64% |
| Men | 98.2% | 0.29% |
| Under 30 | 99.0% | 0.59% |
| 30–44 | 94.2% | 0.59% |
| 45–59 | 94.7% | 0.50% |
| 60+ | 97.3% | 0.22% |

**Known issue:** women's genuine transactions are falsely flagged about 2.2 times as often as men's, and customers under 60 about 2.5 times as often as those over 60. Gender is not an input, so this is proxy bias through other features. Before any real use: identify the driving features, evaluate mitigations (such as reweighting) with compliance, and monitor these rates in production.

## Limitations

- **Synthetic data.** Real-world fraud is messier; expect lower performance on real transactions.
- **Distance from home carried no signal** in this dataset, although it often matters in practice.
- **Labels arrive late** in real life (chargebacks, investigations), so recall can only be measured in arrears.
- **The fraud rate fell** from 0.58% in training to 0.38% in the test months, so precision is lower than it would be at the training rate.
- **Cost assumptions** ($5 per false alarm) are illustrative; a real threshold would use the bank's own costs.

## Monitoring and retraining

- Drift is measured monthly in Databricks (`notebooks/05_drift_monitoring.py`) with the Population Stability Index for each input and for the model score. SageMaker Model Monitor is not used because serverless endpoints do not support data capture.
- **Retrain** when the score or one of the five most important inputs has PSI above 0.25 in any month, or when monthly recall falls below 0.90 once labels arrive.
- A retrained model is registered as a new version; it replaces this one only by moving the `champion` alias after it beats this version on the same held-out evaluation.

## Human oversight

Every alert goes to an analyst. The AI assistant (`agent/`) explains the model's score using only the reasons the endpoint returns, and every recommendation and analyst decision is logged.
