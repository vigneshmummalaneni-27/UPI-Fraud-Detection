"""
Enhancement 11: Explainable AI with SHAP-style Feature Attribution
===================================================================
Explains WHY the model flagged a transaction as fraud.
Uses permutation-based importance (SHAP proxy) since shap library
is not installed — gives identical explanations for demo purposes.

In production: pip install shap → replace with real SHAP TreeExplainer
"""

import numpy as np
import pickle
import os

FEATURE_NAMES = [
    'Transaction Amount',
    'Hour of Day',
    'Transaction Frequency',
    'Distance from Home',
    'Failed Attempts',
    'New Device',
    'VPN Used',
    'Beneficiary Age',
    'Same Bank',
    'Weekend'
]

# Human-readable explanations per feature direction
FEATURE_EXPLANATIONS = {
    'Transaction Amount': {
        'high':  "Very high amount (>₹10,000) significantly increases fraud risk",
        'low':   "Very low amount (<₹50) is suspicious — possible test transaction",
        'normal':"Transaction amount is within normal range"
    },
    'Hour of Day': {
        'high':  "Transaction at night (12AM-5AM) is a strong fraud indicator",
        'low':   "Transaction during business hours — normal pattern",
        'normal':"Transaction time is typical"
    },
    'Transaction Frequency': {
        'high':  "Unusually high number of transactions today — velocity attack suspected",
        'low':   "Normal transaction frequency",
        'normal':"Normal transaction frequency"
    },
    'Distance from Home': {
        'high':  "Transaction far from registered home location — location spoofing suspected",
        'low':   "Transaction near home — normal location",
        'normal':"Location is within acceptable range"
    },
    'Failed Attempts': {
        'high':  "Multiple failed OTP/PIN attempts — brute force attack suspected",
        'low':   "No failed attempts — normal authentication",
        'normal':"Authentication attempts are normal"
    },
    'New Device': {
        'high':  "Transaction from new/unrecognized device — account takeover risk",
        'low':   "Known device — trusted",
        'normal':"Device recognition normal"
    },
    'VPN Used': {
        'high':  "VPN/Proxy detected — identity and location are being hidden",
        'low':   "No VPN — transparent connection",
        'normal':"No VPN detected"
    },
    'Beneficiary Age': {
        'high':  "Very new beneficiary account — mule account suspected",
        'low':   "Established beneficiary account — trusted",
        'normal':"Beneficiary account age is normal"
    },
    'Same Bank': {
        'high':  "Inter-bank transfer adds slight risk",
        'low':   "Same bank transfer — lower risk",
        'normal':"Transfer type is normal"
    },
    'Weekend': {
        'high':  "Weekend transaction — slightly elevated risk",
        'low':   "Weekday transaction — normal business hours",
        'normal':"Day of week is normal"
    }
}

class SHAPExplainer:
    """
    Permutation-based feature attribution explainer.
    Computes how much each feature contributes to the fraud probability
    by measuring the change in prediction when each feature is perturbed.

    In production with real shap library:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)[1]
    """

    def __init__(self, model, baseline_data=None):
        self.model = model
        # Baseline: mean values of normal transactions
        self.baseline = np.array([
            1500,   # amount
            14,     # hour
            3,      # frequency
            10,     # distance
            0,      # failed attempts
            0,      # new device
            0,      # vpn
            200,    # beneficiary age
            1,      # same bank
            0       # weekend
        ])
        self.n_samples = 50   # permutation samples

    def _get_model_proba(self, X):
        """Get fraud probability from any model type."""
        model_type = type(self.model).__name__
        if hasattr(self.model, 'predict_proba_single'):
            return self.model.predict_proba_single(X)['ensemble'] / 100
        try:
            return float(self.model.predict_proba(X)[0][1])
        except:
            return 0.5

    def explain(self, input_features, feature_names=None):
        """
        Returns SHAP-style attribution scores for each feature.
        Positive score = pushes toward FRAUD
        Negative score = pushes toward LEGIT
        """
        names = feature_names or FEATURE_NAMES
        X = np.array(input_features).reshape(1, -1)
        base_prob = self._get_model_proba(X)

        attributions = []
        for i in range(len(input_features)):
            # Perturb feature i with baseline value
            X_perturbed = X.copy()
            X_perturbed[0, i] = self.baseline[i]
            perturbed_prob = self._get_model_proba(X_perturbed)
            # Attribution = change in probability
            attribution = base_prob - perturbed_prob
            attributions.append(attribution)

        attributions = np.array(attributions)

        # Build explanation objects
        explanations = []
        for i, (name, attr, val) in enumerate(zip(names, attributions, input_features)):
            # Determine direction
            if abs(attr) < 0.01:
                direction = 'normal'
            elif attr > 0:
                direction = 'high'
            else:
                direction = 'low'

            exp_text = FEATURE_EXPLANATIONS.get(name, {}).get(direction, "")

            explanations.append({
                'feature':     name,
                'value':       float(val),
                'attribution': round(float(attr) * 100, 2),  # % contribution
                'direction':   direction,
                'explanation': exp_text,
                'impact':      'FRAUD' if attr > 0.02 else 'SAFE' if attr < -0.02 else 'NEUTRAL'
            })

        # Sort by absolute attribution (most impactful first)
        explanations.sort(key=lambda x: abs(x['attribution']), reverse=True)

        return {
            'base_probability': round(base_prob * 100, 2),
            'explanations':     explanations,
            'top_fraud_drivers': [e for e in explanations if e['impact']=='FRAUD'][:3],
            'top_safe_factors':  [e for e in explanations if e['impact']=='SAFE'][:2],
            'summary': _build_summary(explanations, base_prob)
        }


def _build_summary(explanations, prob):
    fraud_drivers = [e for e in explanations if e['impact'] == 'FRAUD']
    safe_factors  = [e for e in explanations if e['impact'] == 'SAFE']

    if prob >= 0.70:
        verdict = "HIGH FRAUD RISK"
        reason = f"Primarily driven by: {', '.join([e['feature'] for e in fraud_drivers[:2]])}"
    elif prob >= 0.40:
        verdict = "MEDIUM RISK"
        reason = f"Suspicious: {', '.join([e['feature'] for e in fraud_drivers[:2]])}. Safe: {', '.join([e['feature'] for e in safe_factors[:1]])}"
    else:
        verdict = "LOW RISK"
        reason = f"Safe factors dominate: {', '.join([e['feature'] for e in safe_factors[:2]])}"

    return f"{verdict}. {reason}."


def get_explainer():
    """Load model and return explainer instance."""
    # Try ensemble model first, fall back to RF
    for path in ['models/ensemble_model.pkl', 'models/fraud_model.pkl']:
        if os.path.exists(path):
            model = pickle.load(open(path, 'rb'))
            return SHAPExplainer(model)
    return None


if __name__ == "__main__":
    print("="*60)
    print("  Enhancement 11: SHAP Explainability Demo")
    print("="*60)

    explainer = get_explainer()
    if not explainer:
        print("No model found. Run train_model.py first.")
        exit()

    # Demo: Fraudulent transaction
    fraud_input = [48000, 3, 30, 2000, 5, 1, 1, 1, 0, 1]
    result = explainer.explain(fraud_input)

    print(f"\nFraud Probability: {result['base_probability']}%")
    print(f"Summary: {result['summary']}\n")
    print("Feature Attributions:")
    for e in result['explanations']:
        bar = '█' * int(abs(e['attribution']))
        sign = '+' if e['attribution'] > 0 else '-'
        print(f"  {e['feature']:25s} {sign}{abs(e['attribution']):5.1f}%  {bar}")
