"""
Enhancement 10: LSTM Deep Learning Model
=========================================
Uses a simulated LSTM-style sequential pattern analysis
(pure numpy/sklearn since TensorFlow is not installed).
In production: replace LSTMFraudModel with real Keras LSTM.

How it works:
- Analyzes sequences of transactions per account (not just single txn)
- Detects temporal patterns: rapid succession, night bursts, escalation
- Ensemble with Random Forest for final prediction
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier   # sklearn MLP = deep-learning proxy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from collections import deque

# ── Sequence Feature Extractor (mimics LSTM memory) ───────────────────────
class SequenceFeatureExtractor:
    """
    Extracts temporal features from transaction sequences.
    In a real LSTM this would be the hidden state; here we compute
    hand-crafted temporal statistics that capture the same patterns.
    """
    def __init__(self, window=5):
        self.window = window
        self.account_history = {}   # account_id → deque of recent txns

    def update(self, account_id, txn_features):
        if account_id not in self.account_history:
            self.account_history[account_id] = deque(maxlen=self.window)
        self.account_history[account_id].append(txn_features)

    def extract_sequence_features(self, account_id, current_txn):
        """
        Returns 8 temporal features:
          0. velocity_score    — transactions per hour (recent window)
          1. amount_spike      — current amount vs recent average
          2. night_burst       — fraction of recent txns at night
          3. device_switches   — how many device changes recently
          4. location_variance — distance spread of recent txns
          5. failure_trend     — rising failed attempts?
          6. new_beneficiary_rate — fraction going to new accounts
          7. escalation_score  — is amount escalating over time?
        """
        history = list(self.account_history.get(account_id, []))
        if len(history) < 2:
            # Cold start: use neutral values
            return np.zeros(8)

        h = np.array(history)   # shape: (n, n_features)
        cur = np.array(current_txn)

        velocity       = len(h) / max(1, (h[-1][1] - h[0][1]) % 24 + 1)
        amt_mean       = h[:, 0].mean() + 1e-6
        amount_spike   = cur[0] / amt_mean
        night_burst    = (h[:, 1] < 6).mean()
        device_switches= (np.diff(h[:, 5]) != 0).sum() if len(h) > 1 else 0
        loc_var        = h[:, 3].std() if len(h) > 1 else 0
        fail_trend     = (np.diff(h[:, 4]) > 0).mean() if len(h) > 1 else 0
        new_ben_rate   = (h[:, 7] < 7).mean()
        escalation     = (np.diff(h[:, 0]) > 0).mean() if len(h) > 1 else 0

        return np.array([velocity, amount_spike, night_burst, device_switches,
                         loc_var, fail_trend, new_ben_rate, escalation])


# ── LSTM Fraud Model (sklearn MLP as deep-learning proxy) ─────────────────
class LSTMFraudModel:
    """
    Deep Learning fraud detector.
    Architecture (simulated):
      Input(18) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Output(1, Sigmoid)

    Replace the MLP with actual Keras LSTM when TF is available:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
    """
    def __init__(self):
        self.seq_extractor = SequenceFeatureExtractor(window=5)
        self.scaler = StandardScaler()
        # Deep MLP = proxy for LSTM (same input/output contract)
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.is_trained = False

    def _build_features(self, X, account_ids=None):
        """Combine raw features with sequence features."""
        seq_feats = []
        for i, row in enumerate(X):
            acc_id = account_ids[i] if account_ids is not None else f"acc_{i}"
            sf = self.seq_extractor.extract_sequence_features(acc_id, row)
            self.seq_extractor.update(acc_id, row)
            seq_feats.append(sf)
        seq_feats = np.array(seq_feats)
        return np.hstack([X, seq_feats])   # 10 raw + 8 sequence = 18 features

    def train(self, X_train, y_train, account_ids=None):
        print("  [LSTM] Building sequence features...")
        X_combined = self._build_features(X_train, account_ids)
        X_scaled = self.scaler.fit_transform(X_combined)
        print("  [LSTM] Training deep network (128→64→32)...")
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        print("  [LSTM] Training complete!")

    def predict_proba(self, X, account_id="unknown"):
        if not self.is_trained:
            return np.array([[0.5, 0.5]])
        sf = self.seq_extractor.extract_sequence_features(account_id, X[0])
        X_combined = np.hstack([X, sf.reshape(1, -1)])
        X_scaled = self.scaler.transform(X_combined)
        return self.model.predict_proba(X_scaled)

    def predict(self, X, account_id="unknown"):
        proba = self.predict_proba(X, account_id)
        return (proba[:, 1] >= 0.5).astype(int)


# ── Ensemble Model (RF + GBM + LSTM-proxy) ────────────────────────────────
class EnsembleFraudDetector:
    """
    Weighted ensemble of 3 models:
      40% Random Forest  (handles tabular features well)
      35% Gradient Boost (captures non-linear interactions)
      25% LSTM-proxy     (captures temporal patterns)
    """
    def __init__(self):
        self.rf   = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.gbm  = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.lstm = LSTMFraudModel()
        self.weights = [0.40, 0.35, 0.25]
        self.rf_scaler = StandardScaler()

    def train(self, X_train, y_train):
        print("\n[Ensemble] Training Random Forest (40% weight)...")
        self.rf.fit(X_train, y_train)

        print("[Ensemble] Training Gradient Boosting (35% weight)...")
        self.gbm.fit(X_train, y_train)

        print("[Ensemble] Training LSTM Deep Network (25% weight)...")
        self.lstm.train(X_train, y_train)

    def predict_proba_single(self, X, account_id="unknown"):
        p_rf  = self.rf.predict_proba(X)[0][1]
        p_gbm = self.gbm.predict_proba(X)[0][1]
        p_lstm= self.lstm.predict_proba(X, account_id)[0][1]
        weighted = (self.weights[0]*p_rf +
                    self.weights[1]*p_gbm +
                    self.weights[2]*p_lstm)
        return {
            'ensemble':  round(float(weighted)*100, 2),
            'random_forest': round(float(p_rf)*100, 2),
            'gradient_boost': round(float(p_gbm)*100, 2),
            'lstm_deep': round(float(p_lstm)*100, 2),
        }

    def predict_proba(self, X):
        p_rf  = self.rf.predict_proba(X)[:, 1]
        p_gbm = self.gbm.predict_proba(X)[:, 1]
        # For batch prediction, lstm uses neutral seq features
        p_lstm = np.array([self.lstm.predict_proba(X[i:i+1])[0][1]
                           for i in range(len(X))])
        weighted = (self.weights[0]*p_rf +
                    self.weights[1]*p_gbm +
                    self.weights[2]*p_lstm)
        return np.column_stack([1-weighted, weighted])


# ── Train and save ─────────────────────────────────────────────────────────
def train_lstm_model():
    print("="*60)
    print("  Enhancement 10: LSTM + Ensemble Model Training")
    print("="*60)

    if not os.path.exists('data/upi_transactions.csv'):
        print("Dataset missing. Run generate_data.py first.")
        return

    df = pd.read_csv('data/upi_transactions.csv')
    FEATURES = ['amount','hour','transaction_frequency','distance_from_home',
                'failed_attempts','new_device','vpn_used','beneficiary_age_days',
                'same_bank','weekend']
    X = df[FEATURES].values
    y = df['is_fraud'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    ensemble = EnsembleFraudDetector()
    ensemble.train(X_train, y_train)

    # Evaluate
    y_prob = ensemble.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n[Ensemble Results]")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")

    os.makedirs('models', exist_ok=True)
    pickle.dump(ensemble, open('models/ensemble_model.pkl', 'wb'))
    print("\n✅ Saved: models/ensemble_model.pkl")
    return ensemble


if __name__ == "__main__":
    train_lstm_model()
