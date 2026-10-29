# Lab 04 – Phish or Legit?

Web application that classifies a message as **phishing / spam or legitimate**, shows the classifier score,
lets you move the decision threshold live, and explains every decision with per-feature contributions
(TF-IDF value × logistic-regression coefficient).

- **Live app:** https://phish-legit-c9jzk4trjxotniydvi2nc4.streamlit.app/
- **Report:** `Lab04_Report.pdf`

> This tool supports a human decision and does not block messages by itself. The data is synthetic
> (plus a sample of the public UCI SMS Spam Collection, which is *spam*, not phishing).

## Data
| Source | Messages | Positive class |
|---|---|---|
| Synthetic (fictional brands, `04_phish_or_legit/data/make_data.py`) | 2,000 | 400 phish |
| UCI SMS Spam Collection (the file hosted on Kaggle), random sample, phone numbers masked | 1,500 | 300 spam |
| **Total** | **3,500** | **700 (20 %)** |

`python 04_phish_or_legit/data/build_augmented.py` rebuilds `messages.csv` from the two sources (seed 42).

## Method
Stratified 75/25 split (seed 42) *before* any fitting → `TfidfVectorizer` (word 1–2-grams) + Naive Bayes or
Logistic Regression inside one `Pipeline` fitted on the training set only. The decision threshold is chosen on
**out-of-fold training scores** (never on the test set).

## Run locally
```bash
cd 04_phish_or_legit
pip install -r requirements.txt
python -m src.evaluate        # metrics, failures, plots -> results/
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
Repository `toubani12/phish-legit`, main file path `04_phish_or_legit/app.py`.

## Limits
Synthetic and template-based data; SMS spam is not phishing; real phishing evolves, so real-world performance
would be lower. Retrain regularly.
