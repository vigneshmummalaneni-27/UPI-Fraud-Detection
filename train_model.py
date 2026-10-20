"""
UPI Fraud Detection - Model Training Script
Run this after generate_data.py
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score, f1_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  UPI FRAUD DETECTION - MODEL TRAINING")
print("=" * 60)

if not os.path.exists('data/upi_transactions.csv'):
    print("Dataset not found. Running data generator...")
    import generate_data
    generate_data.generate_upi_dataset()

df = pd.read_csv('data/upi_transactions.csv')
print(f"\nLoaded {len(df)} transactions")

# ── Feature Engineering ────────────────────────────────────────────────────
FEATURES = ['amount', 'hour', 'transaction_frequency', 'distance_from_home',
            'failed_attempts', 'new_device', 'vpn_used', 'beneficiary_age_days',
            'same_bank', 'weekend']

X = df[FEATURES]
y = df['is_fraud']

print(f"\nClass distribution:")
print(f"  Legitimate: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"  Fraud:      {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

# ── Train/Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── SMOTE for class imbalance ──────────────────────────────────────────────
print("\nApplying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {len(X_train_sm)} training samples")

# ── Scaling ────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sm)
X_test_sc  = scaler.transform(X_test)

# ── Train Models ───────────────────────────────────────────────────────────
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

results = {}
best_model_name = None
best_f1 = 0

print("\nTraining models...\n")
for name, model in models.items():
    print(f"Training {name}...")
    if name == 'Logistic Regression':
        model.fit(X_train_sc, y_train_sm)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
    else:
        model.fit(X_train_sm, y_train_sm)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)

    results[name] = {'accuracy': acc, 'f1': f1, 'auc': auc, 'cm': cm, 'model': model}

    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Confusion Matrix:\n    TN={cm[0,0]}  FP={cm[0,1]}\n    FN={cm[1,0]}  TP={cm[1,1]}\n")

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name

# ── Save Best Model ────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
best_model = results[best_model_name]['model']

pickle.dump(best_model, open('models/fraud_model.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
pickle.dump(FEATURES, open('models/features.pkl', 'wb'))

# Save all results for app
pickle.dump(results, open('models/all_results.pkl', 'wb'))

print("=" * 60)
print(f"  BEST MODEL: {best_model_name}")
print(f"  F1-Score  : {best_f1:.4f}")
print(f"  Saved to  : models/fraud_model.pkl")
print("=" * 60)
print("\nTraining complete! Run app.py to start the web application.")
