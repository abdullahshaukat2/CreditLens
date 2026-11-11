# CreditLens build guide (about 3.5 to 4 hours to a live app)

Goal: a deployed Streamlit app that predicts credit-card default, explains each prediction with SHAP,
and lets you try "what if" changes. You write every line. Hints show the library call you need,
not the full answer. Stuck for more than 10 minutes? Ask Claude for a bigger hint.

| Step | What you build | Time | File |
|---|---|---|---|
| 0 | Setup + GitHub repo | 15 min | `requirements.txt`, `.gitignore` |
| 1 | Explore the data | 25 min | `explore.py` |
| 2 | Train/test split | 15 min | `train.py` |
| 3 | Logistic regression baseline | 20 min | `train.py` |
| 4 | XGBoost | 25 min | `train.py` |
| 5 | Engineered features | 40 min | `features.py` |
| 6 | Save the model + SHAP | 20 min | `train.py` |
| 7 | Streamlit app | 50 min | `app.py` |
| 8 | Deploy to Streamlit Cloud | 20 min | (GitHub) |

Keep a `NOTES.md` as you go: one line per step on what you did and what number you got.
That file becomes your interview script.

---

## Step 0: Setup (15 min)

1. Create `requirements.txt` with: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `streamlit`, `matplotlib`.
2. Create `.gitignore` with `__pycache__/` and `.venv/`.
3. On GitHub, make a new public repo called `creditlens`. In this folder: `git init`, `git add .`,
   `git commit -m "Initial setup"`, `git branch -M main`, `git remote add origin <repo url>`, `git push -u origin main`.
4. Commit after every step from here on. A history of small commits looks good to interviewers.

**Why:** Streamlit Cloud deploys straight from GitHub, so the repo has to exist anyway.

## Step 1: Explore the data (25 min) -> `explore.py`

1. Load `Data/UCI_Credit_Card.csv` (`pd.read_csv`), print `.shape`, `.head()`, `.info()`.
2. Default rate: `df["default.payment.next.month"].mean()`.
3. `df["PAY_0"].value_counts().sort_index()` and compare to `data_labels.md`.
4. Default rate by repayment status: `df.groupby("PAY_0")["default.payment.next.month"].mean()`.
5. Value counts for `EDUCATION` and `MARRIAGE`: find codes the labels don't explain.

**Answer in comments:** What % defaulted, and why does it matter that it isn't 50%? What happens to
default rate as `PAY_0` rises? Which columns would a real lender be careful using (`SEX`, `AGE`, `MARRIAGE`)?

**Interview line:** "About 22% defaulted, so accuracy is misleading: predicting 'no default' for everyone is 78% accurate. That's why I used ROC-AUC."

## Step 2: Train/test split (15 min) -> start `train.py`

1. `X` = every column except `ID` and the target. `y` = the target.
2. `from sklearn.model_selection import train_test_split`
   Split 80/20 with `test_size=0.2`, `stratify=y`, `random_state=42`.
3. Print the sizes (24,000 / 6,000) and the default rate in each part (both about 0.221).

**Why stratify:** it keeps the 22% default rate the same in train and test.
**Why a fixed seed:** anyone who reruns it gets the same split (reproducibility).
**Rule from now on:** the test set is only for the final score. Never tune anything on it.

## Step 3: Logistic regression baseline (20 min)

1. `from sklearn.pipeline import make_pipeline`, `from sklearn.preprocessing import StandardScaler`,
   `from sklearn.linear_model import LogisticRegression`.
2. Build `make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))` and `.fit(X_train, y_train)`.
3. Get probabilities: `model.predict_proba(X_test)[:, 1]`.
4. `from sklearn.metrics import roc_auc_score` and print the AUC. Expect about 0.71.
5. Also print precision, recall, and F1 at a 0.5 cutoff (`(p >= 0.5).astype(int)`).

**Why scale:** logistic regression is sensitive to feature size. `LIMIT_BAL` is in the hundred-thousands and `PAY_0` is -2 to 8.
**Why a baseline:** a fancier model only counts if it beats something simple.
**Know cold:** AUC = the chance the model ranks a random defaulter above a random non-defaulter.

## Step 4: XGBoost (25 min)

1. `from xgboost import XGBClassifier`. Start simple: `XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42)`.
2. Fit on train, score AUC on test. Expect about 0.78.
3. Stretch (optional): tune with `RandomizedSearchCV(..., scoring="roc_auc", cv=5)` on the TRAIN set only.

**Why XGBoost wins:** it's many small decision trees, each fixing the last one's mistakes. It catches
non-linear patterns like "late AND high balance" that a linear model can't.
**No scaling needed:** trees split on thresholds, so feature size doesn't matter.

## Step 5: Engineered features (40 min) -> `features.py`

Write a function `add_features(df)` that returns a copy of `df` with new columns:

| Feature | Formula idea | Why a lender cares |
|---|---|---|
| `util_latest` | `BILL_AMT1 / LIMIT_BAL` | How maxed-out the card is |
| `util_avg_6m` | mean of all 6 bill/limit ratios | Habitual vs one-off high usage |
| `months_late_6m` | count of `PAY_*` columns >= 1 | How often they pay late |
| `max_delay_6m` | max of `PAY_*` columns, floored at 0 | Worst delay |
| `pay_ratio_latest` | `PAY_AMT1 / BILL_AMT2` (careful: divide by zero) | Share of the bill they actually paid |

Hints: `df[cols].div(df["LIMIT_BAL"], axis=0)`, `(df[cols] >= 1).sum(axis=1)`, `np.where(bill > 0, pay / bill, 1.0)`.
Note: `PAY_AMT1` is paid against the *previous* statement, `BILL_AMT2`.

Then in `train.py`, rerun both models with `add_features` applied. Record all four AUCs in `NOTES.md`.

**What you should find:** logistic jumps (about 0.71 to 0.75), XGBoost barely moves (about 0.780 to 0.782).
**Interview gold:** explain why. Trees already find these patterns. Linear models need them handed over.

## Step 6: Save the model + SHAP (20 min)

1. Save the final XGBoost: `model.save_model("model.json")`.
2. Save the test set with predictions: add columns `risk_score` and `actual_default`, then `.to_csv("test_set.csv")`.
3. `import shap`; `explainer = shap.TreeExplainer(model)`; `sv = explainer(X_test)`.
4. Print the mean absolute SHAP per feature to find the top drivers (expect `PAY_0` first).
5. Also print a threshold table: for cutoffs 0.2 to 0.5, what % are flagged, precision, and recall.

**SHAP in one sentence:** for one borrower, how much each feature pushed their risk up or down from the average.
**Commit `model.json` and `test_set.csv`.** The deployed app loads them instead of retraining.

## Step 7: Streamlit app (50 min) -> `app.py`

Build three sections:

1. **Pick a borrower:** load `model.json` (`XGBClassifier(); m.load_model("model.json")`) and `test_set.csv`.
   Use `@st.cache_resource` so it loads once. Add `st.selectbox` of row IDs and `st.metric` for the risk %.
2. **Why this score:** `shap.plots.waterfall(sv[0], max_display=10, show=False)` then `st.pyplot(plt.gcf())`.
3. **What if:** `st.slider` for `PAY_0`, `st.number_input` for `LIMIT_BAL`, and a slider to scale payments.
   Copy the row, change those values, rerun `add_features`, predict, and show the new risk with a `delta`.

Run locally with `python -m streamlit run app.py`.

## Step 8: Deploy (20 min)

1. Push everything (including `model.json`, `test_set.csv`, and `Data/`) to GitHub.
2. Go to share.streamlit.io, sign in with GitHub, click "Create app", and pick your repo, branch `main`, and file `app.py`.
3. Under Advanced settings choose Python 3.12. Some libraries may not support 3.14 on the server yet.
4. Wait for the build, then open the URL. Add it to your GitHub README and resume.
5. Free apps sleep after inactivity. Open it the morning of each interview.

---

## Done checklist

- [ ] All four AUC numbers in `NOTES.md`, from your own run
- [ ] You can explain every feature in `features.py` without looking
- [ ] README with the results table, the live link, and one screenshot
- [ ] App is live and loads in under a minute
