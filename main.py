import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

# Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Convert target to numeric
y = train["Heart Disease"].map({"Absence": 0, "Presence": 1}).astype(int)

# Remove target from features
X = train.drop(columns=["Heart Disease"])

# Save test IDs for later submission
test_ids = test["id"]

# Remove id column (not useful for model)
X = X.drop(columns=["id"])
X_test = test.drop(columns=["id"])

# Split the data
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Logistic Regression model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=2000))
])

# Train the model
model.fit(X_train, y_train)

# Predict probabilities (IMPORTANT: use predict_proba)
val_pred = model.predict_proba(X_val)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_val, val_pred)

print("\nBaseline Logistic Regression AUC:", auc)


# Create stronger model
hgb_model = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=6,
    max_leaf_nodes=31,
    min_samples_leaf=30,
    max_iter=300,
    random_state=42
)

# Train model
hgb_model.fit(X_train, y_train)

# Predict probabilities
val_pred_hgb = hgb_model.predict_proba(X_val)[:, 1]

# Calculate AUC
auc_hgb = roc_auc_score(y_val, val_pred_hgb)

print("\nHistGradientBoosting AUC:", auc_hgb)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

aucs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        random_state=42
    )

    model.fit(X_tr, y_tr)
    pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pred)

    aucs.append(auc)
    print(f"Fold {fold+1} AUC:", auc)

print("\nMean CV AUC:", np.mean(aucs))


final_model = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=6,
    max_leaf_nodes=31,
    min_samples_leaf=30,
    max_iter=300,
    random_state=42
)

# Train on FULL dataset
final_model.fit(X, y)

test_predictions = final_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "id": test_ids,
    "Heart Disease": test_predictions
})

submission.to_csv("submission.csv", index=False)

print("Submission file created successfully.")

