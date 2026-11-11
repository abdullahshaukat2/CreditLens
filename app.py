"""CreditLens: credit-card default risk explorer.

Run locally:  python -m streamlit run app.py
Needs model.json and test_set.csv from train.py, plus features.py.
"""
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from xgboost import XGBClassifier

from features import add_features

st.set_page_config(page_title="CreditLens", page_icon="💳", layout="wide")

# Readable names for the features shown to users
LABELS = {
    "LIMIT_BAL": "Credit limit", "SEX": "Sex", "EDUCATION": "Education", "MARRIAGE": "Marital status",
    "AGE": "Age", "PAY_0": "Repayment status (Sep)", "PAY_2": "Repayment status (Aug)",
    "PAY_3": "Repayment status (Jul)", "PAY_4": "Repayment status (Jun)",
    "PAY_5": "Repayment status (May)", "PAY_6": "Repayment status (Apr)",
    "BILL_AMT1": "Bill (Sep)", "BILL_AMT2": "Bill (Aug)", "BILL_AMT3": "Bill (Jul)",
    "BILL_AMT4": "Bill (Jun)", "BILL_AMT5": "Bill (May)", "BILL_AMT6": "Bill (Apr)",
    "PAY_AMT1": "Payment (Sep)", "PAY_AMT2": "Payment (Aug)", "PAY_AMT3": "Payment (Jul)",
    "PAY_AMT4": "Payment (Jun)", "PAY_AMT5": "Payment (May)", "PAY_AMT6": "Payment (Apr)",
    "util_latest": "Utilization (Sep)", "util_avg_6m": "Avg utilization (6 mo)",
    "util_max_6m": "Max utilization (6 mo)", "months_late_6m": "Months late (6 mo)",
    "max_delay_6m": "Worst delay (months)", "pay_ratio_latest": "Share of last bill paid",
}
STATUS_TEXT = {-2: "No balance", -1: "Paid in full", 0: "Paid minimum (revolving)"}

# Results from train.py (same split, seed 42). Update if you retrain.
MODEL_RESULTS = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost", "XGBoost (tuned)"],
    "AUC, raw features": [0.708, 0.775, 0.778, 0.780],
    "AUC, engineered features": [0.746, 0.776, 0.780, 0.781],
})


@st.cache_resource
def load():
    model = XGBClassifier()
    model.load_model("model.json")
    test = pd.read_csv("test_set.csv", index_col=0)
    return model, test, shap.TreeExplainer(model)


@st.cache_data
def global_importance(_explainer, X):
    sv = _explainer(X)
    imp = pd.Series(np.abs(sv.values).mean(axis=0), index=X.columns)
    return imp.sort_values(ascending=False)


def status_text(v):
    v = int(v)
    return STATUS_TEXT.get(v, f"{v} month{'s' if v > 1 else ''} late")


# ---------------- Light / dark mode for charts ----------------
# Streamlit has built-in Light / Dark / System themes (menu ⋮ -> Settings).
# Matplotlib charts don't follow it automatically, so we restyle them here.
DARK = st.context.theme.type == "dark"
FG = "#FAFAFA" if DARK else "#262730"
GRID = "#4A4F5C" if DARK else "#D0D3DA"
ACCENT = "#4DA3FF" if DARK else "#1F77B4"
plt.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": GRID,
    "xtick.color": FG, "ytick.color": FG, "legend.facecolor": "none", "legend.edgecolor": GRID,
})

model, test, explainer = load()
FEATURES = model.get_booster().feature_names
y_true = test["actual_default"].to_numpy()
scores = test["risk_score"].to_numpy()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("💳 CreditLens")
    st.write("Predicts whether a credit-card borrower will **default next month**, "
             "and explains why.")
    cutoff = st.slider("Flag borrowers at or above this risk", 0.05, 0.95, 0.30, 0.05,
                       help="Lower = catch more defaulters but flag more good customers.")
    st.caption("🌗 Light / dark mode: menu ⋮ (top right) → Settings → Theme.")
    st.caption(f"Model: tuned XGBoost · Test AUC **{roc_auc_score(y_true, scores):.3f}**")
    st.caption("Data: UCI Default of Credit Card Clients (Taiwan, 2005). "
               "Scores shown are on 6,000 borrowers the model never trained on.")


def risk_band(p):
    if p >= cutoff:
        return "🔴 Flagged"
    if p >= cutoff / 2:
        return "🟠 Watch"
    return "🟢 Low risk"


st.title("CreditLens")
st.caption("Credit-card default risk, explained borrower by borrower.")

tab1, tab2, tab3 = st.tabs(["🔍 Borrower lookup", "🧪 What-if simulator", "📊 Model performance"])

# ---------------- Tab 1: Borrower lookup ----------------
with tab1:
    c1, c2 = st.columns([1, 3])
    with c1:
        order = st.radio("Show", ["Highest risk", "Lowest risk", "Random"], key="order")
        if order == "Random":
            pool = test.sample(200, random_state=7)
        else:
            pool = test.sort_values("risk_score", ascending=(order == "Lowest risk")).head(200)
        borrower = st.selectbox("Borrower ID", pool.index, key="borrower")
    row = test.loc[[borrower]]
    r = row.iloc[0]
    risk = float(r["risk_score"])

    with c2:
        m1, m2, m3, m4 = st.columns([1, 1.4, 1.1, 1.1])
        m1.metric("Predicted risk", f"{risk:.0%}")
        m2.metric("Decision", risk_band(risk))
        m3.metric("Actually defaulted", "Yes" if r["actual_default"] == 1 else "No")
        m4.metric("Credit limit", f"NT${r['LIMIT_BAL'] / 1000:,.0f}K")
        st.progress(min(risk, 1.0))

    left, right = st.columns([1, 1.4])
    with left:
        st.subheader("Profile")
        prof = pd.DataFrame({
            "Month": ["Sep", "Aug", "Jul", "Jun", "May", "Apr"],
            "Bill (NT$)": [r[f"BILL_AMT{i}"] for i in range(1, 7)],
            "Paid (NT$)": [r[f"PAY_AMT{i}"] for i in range(1, 7)],
            "Status": [status_text(r[c]) for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]],
        })
        st.dataframe(prof, hide_index=True, width="stretch",
                     column_config={"Bill (NT$)": st.column_config.NumberColumn(format="%d"),
                                    "Paid (NT$)": st.column_config.NumberColumn(format="%d")})
        st.write(f"**Months late in last 6:** {int(r['months_late_6m'])} · "
                 f"**Max utilization:** {r['util_max_6m']:.0%} · "
                 f"**Share of last bill paid:** {r['pay_ratio_latest']:.0%}")

    with right:
        st.subheader("Why this score")
        sv = explainer(row[FEATURES])
        contrib = pd.Series(sv.values[0], index=FEATURES)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(3)
        for f, v in top.items():
            arrow = "raises" if v > 0 else "lowers"
            st.write(f"- **{LABELS.get(f, f)}** {arrow} risk")
        exp = shap.Explanation(sv.values[0], sv.base_values[0], row[FEATURES].iloc[0].values,
                               feature_names=[LABELS.get(f, f) for f in FEATURES])
        shap.plots.waterfall(exp, max_display=8, show=False)
        fig = plt.gcf()
        fig.set_size_inches(7, 4.2)
        st.pyplot(fig, clear_figure=True)
        st.caption("Starts at the average borrower's risk (bottom) and adds each feature's push. "
                   "Red raises risk, blue lowers it. Units are log-odds.")

# ---------------- Tab 2: What-if ----------------
with tab2:
    st.write(f"Change borrower **#{borrower}**'s profile and see how the model reacts. "
             "Engineered features (utilization, months late, etc.) are recomputed automatically.")
    a, b, c, d = st.columns(4)
    new_limit = a.number_input("Credit limit (NT$)", 10000, 1_000_000, int(r["LIMIT_BAL"]), step=10000)
    new_status = b.select_slider("September repayment status", options=list(range(-2, 9)),
                                 value=int(r["PAY_0"]), format_func=status_text)
    pay_scale = c.slider("Multiply all payments by", 0.0, 3.0, 1.0, 0.1)
    bill_scale = d.slider("Multiply all bills by", 0.0, 2.0, 1.0, 0.1)

    sim = row[FEATURES].copy()
    sim["LIMIT_BAL"] = new_limit
    sim["PAY_0"] = new_status
    for i in range(1, 7):
        sim[f"PAY_AMT{i}"] = sim[f"PAY_AMT{i}"] * pay_scale
        sim[f"BILL_AMT{i}"] = sim[f"BILL_AMT{i}"] * bill_scale
    sim = add_features(sim)[FEATURES]
    new_risk = float(model.predict_proba(sim)[0, 1])

    k1, k2, k3 = st.columns(3)
    k1.metric("Original risk", f"{risk:.0%}")
    k1.caption(risk_band(risk))
    k2.metric("New risk", f"{new_risk:.0%}", f"{(new_risk - risk) * 100:+.1f} pts", delta_color="inverse")
    k3.metric("New decision", risk_band(new_risk))
    st.progress(min(new_risk, 1.0))

# ---------------- Tab 3: Model performance ----------------
with tab3:
    pred = (scores >= cutoff).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()
    st.subheader(f"At a {cutoff:.2f} cutoff")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Borrowers flagged", f"{pred.mean():.1%}")
    p2.metric("Defaulters caught (recall)", f"{tp / (tp + fn):.0%}", f"{tp:,} of {tp + fn:,}", delta_color="off")
    p3.metric("Flags that were right (precision)", f"{tp / max(tp + fp, 1):.0%}")
    p4.metric("Good customers flagged", f"{fp:,}")

    g1, g2, g3 = st.columns(3)
    with g1:
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1], ["Not flagged", "Flagged"])
        ax.set_yticks([0, 1], ["Paid", "Defaulted"])
        ax.set_title("Confusion matrix")
        st.pyplot(fig, clear_figure=True)
    with g2:
        fpr, tpr, thr = roc_curve(y_true, scores)
        idx = np.argmin(np.abs(thr - cutoff))
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.plot(fpr, tpr, lw=2, color=ACCENT, label=f"XGBoost (AUC {roc_auc_score(y_true, scores):.3f})")
        ax.plot([0, 1], [0, 1], "--", color="grey", label="Coin flip (0.5)")
        ax.scatter(fpr[idx], tpr[idx], color="red", zorder=3, label="Current cutoff")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate (recall)")
        ax.set_title("ROC curve")
        ax.legend(fontsize=8, loc="lower right")
        st.pyplot(fig, clear_figure=True)
    with g3:
        imp = global_importance(explainer, test[FEATURES]).head(8)[::-1]
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.barh([LABELS.get(f, f) for f in imp.index], imp.values, color=ACCENT)
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title("Top drivers (all test borrowers)")
        st.pyplot(fig, clear_figure=True)

    st.subheader("Model comparison (ROC-AUC on the same 6,000-borrower test set)")
    st.dataframe(MODEL_RESULTS, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in MODEL_RESULTS.columns[1:]})
    st.caption("Engineered features lifted logistic regression from 0.708 to 0.746, but barely moved the "
               "tree models, which already learn those patterns from raw columns.")
