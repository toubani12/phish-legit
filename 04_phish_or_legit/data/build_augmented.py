"""Lab 04 - augment the synthetic phishing dataset with the real SMS Spam Collection (UCI / Kaggle).

Run from the lab folder:  python data/build_augmented.py   ->  data/messages.csv

Inputs (kept in the repo so the build is reproducible):
  data/messages_synthetic.csv      2,000 synthetic messages (400 phish / 1,600 legit), from make_data.py
  data/sms_spam_collection.tsv     UCI SMS Spam Collection (the file published on Kaggle as
                                   "SMS Spam Collection Dataset"), 5,574 real SMS: ham / spam

Honesty note: the SMS collection is SPAM (lottery, premium-rate, promotions), not phishing. Both are mapped
to the positive class "phish" = "unwanted/malicious message", and every row keeps a `source` column so
results can be reported per source.  A random sample keeps the 20 % prevalence (300 spam + 1,200 ham).
Privacy: phone numbers (7+ digits) and short codes are masked before the sample is saved.
"""
import re
from pathlib import Path

import pandas as pd

SEED = 42
N_SPAM, N_HAM = 300, 1200
HERE = Path(__file__).resolve().parent
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d \-]{5,}\d(?!\w)")


def clean(text: str) -> str:
    text = PHONE_RE.sub("<phone>", str(text))
    return re.sub(r"\s+", " ", text).strip()


def main():
    syn = pd.read_csv(HERE / "messages_synthetic.csv")
    syn["source"] = "synthetic"

    sms = pd.read_csv(HERE / "sms_spam_collection.tsv", sep="\t", header=None, names=["label", "text"])
    sms = sms.drop_duplicates("text")
    sms["label"] = sms["label"].map({"spam": "phish", "ham": "legit"})
    sms["text"] = sms["text"].map(clean)
    sample = pd.concat([sms[sms.label == "phish"].sample(N_SPAM, random_state=SEED),
                        sms[sms.label == "legit"].sample(N_HAM, random_state=SEED)])
    sample["source"] = "sms_spam_uci"

    out = pd.concat([syn, sample[["text", "label", "source"]]], ignore_index=True)
    out = out.sample(frac=1, random_state=SEED).reset_index(drop=True)
    out.to_csv(HERE / "messages.csv", index=False)
    print(out.groupby(["source", "label"]).size())
    print("total", len(out), "| phish rate", round((out.label == "phish").mean(), 3))


if __name__ == "__main__":
    main()
