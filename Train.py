from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap

#linear regression model imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

#Random forest model imports
from sklearn.ensemble import RandomForestClassifier

#Xgboost model imports
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from features import add_features


SEED = 42
TARGET = "default.payment.next.month"
df = pd.read_csv('Data/UCI_Credit_Card.csv')
df = add_features(df)
x = df.drop(columns=['ID',TARGET])
y= df[TARGET]
print(x.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y, random_state=SEED)

print("Train:", X_train.shape, "default rate:", round(y_train.mean(), 3))
print("Test: ", X_test.shape, "default rate:", round(y_test.mean(), 3))

# logistic regression model

logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
logreg.fit(X_train, y_train)

p_logreg = logreg.predict_proba(X_test)[:, 1]

def evaluate(name, p, y_true):
    pred = (p >= 0.5).astype(int)
    print(f"\n===== {name} =====")
    print(f"AUC: {roc_auc_score(y_true, p):.3f}")
    print(classification_report(y_true, pred, digits=3))

    cm = confusion_matrix(y_true, pred)
    print(f"Caught {cm[1, 1]} of {cm[1].sum()} defaulters, missed {cm[1, 0]}")

    ConfusionMatrixDisplay(cm, display_labels=["Paid", "Defaulted"]).plot(cmap="Blues")
    plt.title(name)
    plt.savefig(f"cm_{name.replace(' ', '_').lower()}.png", bbox_inches="tight")
    plt.close()


evaluate("Logistic Regression", p_logreg, y_test)

#Random Forest model

rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=SEED)
rf.fit(X_train, y_train)
p_rf = rf.predict_proba(X_test)[:, 1]
evaluate("Random Forest", p_rf, y_test)


#XGBoost model
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=SEED)
xgb.fit(X_train, y_train)
p_xgb = xgb.predict_proba(X_test)[:, 1]
evaluate("XGBoost", p_xgb, y_test)

#Commeneted out afetr finding the best hyperparameters for XGBoost model using RandomizedSearchCV. 
#The best hyperparameters are used to create a new XGBoost model and evaluate it.

# params = {
#     "n_estimators": [200, 300, 500],
#     "max_depth": [3, 4, 5, 6],
#     "learning_rate": [0.02, 0.05, 0.1],
#     "subsample": [0.7, 0.8, 1.0],
#     "colsample_bytree": [0.6, 0.8, 1.0],
#     "min_child_weight": [1, 5, 10],
# }
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
# search = RandomizedSearchCV(
#     XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1),
#     params, n_iter=30, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=-1,
# )
# search.fit(X_train, y_train)
# print("Best settings:", search.best_params_)
# print(f"Cross-validated AUC: {search.best_score_:.3f}")

# xgb_tuned = search.best_estimator_
# p_xgb_tuned = xgb_tuned.predict_proba(X_test)[:, 1]
# evaluate("XGBoost Tuned", p_xgb_tuned, y_test)

xgb_tuned = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.02, subsample=0.8,
                          colsample_bytree=0.6, min_child_weight=5,
                          eval_metric="logloss", random_state=SEED, n_jobs=-1)

xgb_tuned.fit(X_train, y_train)
p_xgb_tuned = xgb_tuned.predict_proba(X_test)[:, 1]
evaluate("XGBoost Tuned", p_xgb_tuned, y_test)


xgb_tuned.save_model("model.json")

test_out = X_test.copy()
test_out["actual_default"] = y_test
test_out["risk_score"] = p_xgb_tuned
test_out.to_csv("test_set.csv")
print("Saved model.json and test_set.csv")

explainer = shap.TreeExplainer(xgb_tuned)
sv = explainer(X_test)
importance = pd.Series(np.abs(sv.values).mean(axis=0), index=X_test.columns)
print(importance.sort_values(ascending=False).head(10))
shap.plots.beeswarm(sv, max_display=12, show=False)
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.close()

print("\nCutoff trade-off (XGBoost Tuned):")
for cutoff in [0.2, 0.3, 0.4, 0.5]:
        pred = (p_xgb_tuned >= cutoff).astype(int)
        print(f"cutoff {cutoff}: flags {pred.mean():.1%} of borrowers, "
        f"precision {precision_score(y_test, pred):.3f}, recall {recall_score(y_test, pred):.3f}")







