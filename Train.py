from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

#linear regression model imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


TARGET = "default.payment.next.month"
df = pd.read_csv('Data/UCI_Credit_Card.csv')
x = df.drop(columns=['ID',TARGET])
y= df[TARGET]
print(x.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y, random_state=42)

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







