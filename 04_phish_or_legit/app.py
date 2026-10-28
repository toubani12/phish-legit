"""Lab 04 - Phish or Legit? (Streamlit UI). Run:  streamlit run app.py"""
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from src.pipeline import (load_data, split, make_pipeline, predict, word_weights, contributions,
                          find_obfuscation)

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="Phish or Legit?", page_icon="🎣", layout="centered")

EXAMPLES = {
    "🏦 Bank phishing": "Northbank: your account has been locked. Verify your identity now http://bit.ly/ab12cd "
                        "Within 24 hours or your access will be suspended.",
    "💰 Lottery scam": "Ref No: W B C F/XGM/01015/0026\n\nYour e-m a i l address was chosen at random during an "
                       "internet search to receive USD2,805,000.00 from me, the chairman and CEO of Berkshire "
                       "Hathaway. If you are i n t e r e s t e d, to learn more about the donation and how to "
                       "c l a i m it.\n\nRegards,\nEdward Buffe",
    "💬 Friendly chat": "Are we still on for dinner at 7:30? Running late, see you soon.",
    "🧾 Legit invoice": "Your invoice of $65.47 is ready. Details are in your online account.",
}
FRIENDLY = {"urltok": "a link", "numtok": "a number", "moneytok": "a money amount"}


@st.cache_resource(show_spinner="Training the model on first load...")
def get_model():
    df = load_data()
    Xtr, _, ytr, _ = split(df)
    return make_pipeline("lr").fit(Xtr, ytr), make_pipeline("lr_cal").fit(Xtr, ytr)


def level(score, thr):
    if score >= thr:
        return "Phishing", "🚨", "red"
    if score >= thr / 2:
        return "Suspicious", "⚠️", "orange"
    return "Looks safe", "✅", "green"


def highlight(text, weights):
    parts = []
    for tok in re.findall(r"\w+|\W+", text):
        w = weights.get(tok.lower(), 0.0)
        if re.match(r"\w+", tok) and abs(w) > 0.3:
            parts.append(f":red[**{tok}**]" if w > 0 else f":green[**{tok}**]")
        else:
            parts.append(tok)
    return "".join(parts)


model, model_cal = get_model()

with st.sidebar:
    st.header("Settings")
    thr = st.slider("Phishing threshold", 0.20, 0.90, 0.50, 0.05,
                    help="Lower = catch more scams but more false alarms. Higher = fewer false alarms but more misses.")
    st.caption("Recall vs precision (logistic regression, test set): 0.2 → 0.93 / 0.58 · 0.5 → 0.69 / 0.79 · "
               "0.9 → 0.17 / 1.00")
    st.divider()
    st.caption("Trained on a SYNTHETIC dataset with fictional brands. Labels mean 'phishing-style' vs 'legit', "
               "not real-world ground truth.")

st.title("🎣 Phish or Legit?")
st.caption("TF-IDF + logistic regression · paste a message and see why the model decides what it does.")

tab_cls, tab_perf, tab_about = st.tabs(["Classify", "Model performance", "About"])

with tab_cls:
    st.write("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, (name, ex) in zip(cols, EXAMPLES.items()):
        if col.button(name, width="stretch"):
            st.session_state["msg"] = ex
    st.session_state.setdefault("msg", EXAMPLES["🏦 Bank phishing"])
    text = st.text_area("Paste a message:", key="msg", height=160)
    st.caption(f"{len(text)} characters")

    if st.button("Classify", type="primary"):
        try:
            _, score = predict(model, text)
            _, cal_score = predict(model_cal, text)
        except ValueError as err:
            st.error(str(err))
        else:
            name, icon, color = level(score, thr)
            c1, c2 = st.columns([2, 1])
            c1.markdown(f"## {icon} :{color}[{name}]")
            c2.metric("Phishing score", f"{score:.0%}")
            st.progress(min(score, 1.0))
            with st.expander("About this score"):
                st.write(f"Calibrated score: **{cal_score:.2f}**. Scores are only meaningful as probabilities if "
                         "calibration was checked - see the Model performance tab.")

            hidden = find_obfuscation(text)
            if hidden:
                shown = ", ".join(f"`{a}` → `{b}`" for a, b in hidden[:5])
                st.warning(f"**Hidden-text trick detected.** Spaced-out letters are a common way to dodge spam "
                           f"filters: {shown}")

            contrib = contributions(model, text)
            top = list(contrib.items())[:8]
            if top:
                st.subheader("Why?")
                names = [FRIENDLY.get(t, t) for t, _ in top]
                df = pd.DataFrame({"feature": names, "contribution": [v for _, v in top]}).set_index("feature")
                st.bar_chart(df, horizontal=True)
                st.caption("Positive = pushes toward phishing, negative = toward legit.")
            st.markdown(highlight(text, word_weights(model, text)))
            st.caption("red = pushes toward phishing, green = pushes toward legit (strong words only)")

            st.download_button("Download report", data=json.dumps(
                {"message": text, "score": round(score, 4), "calibrated_score": round(cal_score, 4),
                 "threshold": thr, "verdict": name,
                 "top_features": {t: round(v, 3) for t, v in top}}, indent=2),
                file_name="phish_report.json", mime="application/json")

with tab_perf:
    mfile = HERE / "results" / "metrics.json"
    if mfile.exists():
        data = json.loads(mfile.read_text())
        st.subheader("Models compared")
        st.dataframe(pd.DataFrame(data["models"]).T.round(3), width="stretch")
        st.subheader("Confusion matrix (logistic regression)")
        st.dataframe(pd.DataFrame(data["confusion_matrix_lr"], index=["actual legit", "actual phish"],
                                  columns=["pred legit", "pred phish"]))
    img = HERE / "results" / "phish_results.png"
    if img.exists():
        st.image(str(img), width="stretch")
    if not mfile.exists() and not img.exists():
        st.info("Run `python src/evaluate.py` to generate the results.")

with tab_about:
    st.markdown("""
**How it works.** Messages are normalised (links, money and numbers become placeholders, letter-spaced words
like `c l a i m` are rejoined), turned into TF-IDF features, and scored by logistic regression.

**Limits.** The data is synthetic and built from templates, so real-world scams with different wording can slip
through. Do not use this as a security tool.
""")
