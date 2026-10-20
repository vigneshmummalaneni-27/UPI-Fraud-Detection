"""
Enhanced UPI Fraud Detection App
Integrates Enhancements 10-14:
  10. LSTM Ensemble Model
  11. SHAP Explainability
  12. Real-Time Streaming
  13. Mobile API
  14. GNN Fraud Ring Detection

Run: python enhanced_app.py
     http://localhost:5000
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'enhancements'))

from flask import Flask, render_template, request, jsonify
import pickle, numpy as np, uuid
from datetime import datetime

app = Flask(__name__, template_folder='templates')

# ── Required class for loading ensemble model ──────────────────────────────
class EnsembleFraudDetector:
    def __init__(self, models=None):
        self.models = models

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return max(set(predictions), key=predictions.count)

# ── Load models ────────────────────────────────────────────────────────────
base_model = None
scaler = None

if os.path.exists('models/fraud_model.pkl'):
    base_model = pickle.load(open('models/fraud_model.pkl', 'rb'))
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))

ensemble_model = None
if os.path.exists('models/ensemble_model.pkl'):
    ensemble_model = pickle.load(open('models/ensemble_model.pkl', 'rb'))

results_data = None
if os.path.exists('models/all_results.pkl'):
    raw = pickle.load(open('models/all_results.pkl', 'rb'))
    results_data = {
        name: {
            'accuracy': round(v['accuracy'] * 100, 2),
            'f1': round(v['f1'] * 100, 2),
            'auc': round(v['auc'] * 100, 2),
            'cm': v['cm'].tolist()
        }
        for name, v in raw.items()
    }
# ── Load enhancements ──────────────────────────────────────────────────────
from enhancements.enhancement_11_shap      import SHAPExplainer, FEATURE_NAMES
from enhancements.enhancement_12_streaming import (start_streaming, stop_streaming,
                                                    get_stream_metrics)
from enhancements.enhancement_13_mobile_api import mobile_api
from enhancements.enhancement_14_gnn        import (add_to_graph, get_account_graph_score,
                                                     get_graph_summary)

app.register_blueprint(mobile_api)

# Try LSTM explainer first, fallback to base
_explainer_model = ensemble_model or base_model
explainer = SHAPExplainer(_explainer_model) if _explainer_model else None

# In-memory stores
TRANSACTION_LOG  = []
BLOCKED_ACCOUNTS = {}
FLAGGED_ACCOUNTS = {}
SMS_LOG          = []

# ── SMS simulation ─────────────────────────────────────────────────────────
def send_sms(phone, message, txn_id, msg_type):
    sms = {'id': str(uuid.uuid4())[:8].upper(), 'txn_id': txn_id,
           'phone': phone, 'message': message, 'type': msg_type,
           'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
           'status': 'SENT (Simulated)'}
    SMS_LOG.append(sms)
    return sms

def block_transaction(txn_id, account_id, phone, reason, prob):
    BLOCKED_ACCOUNTS[account_id] = {
        'txn_id': txn_id, 'reason': reason,
        'blocked_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'probability': prob}
    msg = (f"ALERT: Your UPI transaction {txn_id} has been BLOCKED "
           f"due to suspicious activity ({prob:.0f}% fraud risk). "
           f"Call 1800-XXX-XXXX to unblock. -YourBank")
    return {'action':'BLOCKED','txn_id':txn_id,
            'message':f'Transaction BLOCKED. SMS alert sent to {phone}.',
            'sms':send_sms(phone,msg,txn_id,'FRAUD_ALERT')}

def flag_for_review(txn_id, account_id, phone, reason, prob):
    otp = str(np.random.randint(100000,999999))
    FLAGGED_ACCOUNTS[account_id] = {
        'txn_id': txn_id, 'reason': reason,
        'flagged_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'probability': prob, 'status': 'PENDING REVIEW', 'otp': otp}
    msg = (f"Your UPI transaction {txn_id} needs verification. "
           f"OTP: {otp} (valid 5 min). Do NOT share. -YourBank")
    return {'action':'OTP_REQUIRED','txn_id':txn_id,
            'message':f'OTP sent to {phone}. Verification required.',
            'otp':otp,'sms':send_sms(phone,msg,txn_id,'OTP_REQUEST')}

def approve_transaction(txn_id, phone, prob):
    msg = (f"SUCCESS: UPI Transaction {txn_id} approved. "
           f"Risk Score: {prob:.0f}% (LOW). -YourBank")
    return {'action':'APPROVED','txn_id':txn_id,
            'message':'Transaction APPROVED. Confirmation SMS sent.',
            'sms':send_sms(phone,msg,txn_id,'APPROVAL')}

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('enhanced_index.html', results=results_data)

@app.route('/predict', methods=['POST'])
def predict():
    if not base_model:
        return jsonify({'error':'Model not loaded. Run train_model.py first!'}),500
    try:
        data       = request.json
        account_id = data.get('account_id','ACC'+str(np.random.randint(10000,99999)))
        phone      = data.get('phone','+91-98XXXXXXXX')
        txn_id     = 'TXN'+str(uuid.uuid4())[:8].upper()

        input_arr = np.array([[
            float(data['amount']), int(data['hour']),
            int(data['transaction_frequency']), float(data['distance_from_home']),
            int(data['failed_attempts']), int(data['new_device']),
            int(data['vpn_used']), int(data['beneficiary_age_days']),
            int(data['same_bank']), int(data['weekend'])
        ]])

        # Enhancement 10: Use ensemble if available
        model_name = type(base_model).__name__
        if ensemble_model:
            scores = ensemble_model.predict_proba_single(input_arr, account_id)
            prob_pct = scores['ensemble']
            model_scores = scores
            model_name = 'Ensemble (RF+GBM+LSTM)'
        else:
            mt = type(base_model).__name__
            if mt == 'LogisticRegression':
                p = base_model.predict_proba(scaler.transform(input_arr))[0][1]
            else:
                p = base_model.predict_proba(input_arr)[0][1]
            prob_pct = round(float(p)*100,2)
            model_scores = {'ensemble': prob_pct, 'random_forest': prob_pct,
                            'gradient_boost': None, 'lstm_deep': None}

        # Risk factors
        risk_factors = []
        if float(data['amount'])>10000:             risk_factors.append("High transaction amount (>Rs.10,000)")
        if int(data['hour'])<6:                     risk_factors.append("Unusual time - midnight hours (12AM-5AM)")
        if int(data['transaction_frequency'])>15:   risk_factors.append("Abnormally high transaction frequency")
        if float(data['distance_from_home'])>200:   risk_factors.append("Transaction far from registered address")
        if int(data['failed_attempts'])>=3:         risk_factors.append("Multiple failed OTP/PIN attempts")
        if int(data['new_device'])==1:              risk_factors.append("Transaction from new/unrecognized device")
        if int(data['vpn_used'])==1:                risk_factors.append("VPN/Proxy connection detected")
        if int(data['beneficiary_age_days'])<5:     risk_factors.append("Beneficiary account very new (<5 days)")

        # Enhancement 11: SHAP explanation
        shap_result = None
        if explainer:
            try:
                shap_result = explainer.explain(input_arr[0].tolist())
            except:
                pass

        # Enhancement 14: Graph score
        receiver_id = data.get('receiver_id', 'REC'+str(np.random.randint(10000,99999)))
        add_to_graph(account_id, receiver_id, float(data['amount']),
                     features=input_arr[0].tolist(),
                     is_fraud=(prob_pct>=70))
        graph_score = get_account_graph_score(account_id)

        # Combined risk (blend transaction + graph score)
        graph_risk  = graph_score.get('graph_risk', 0)
        final_prob  = round(prob_pct * 0.75 + graph_risk * 0.25, 2)

        if graph_score.get('is_fraud_ring_member'):
            risk_factors.append(f"⚠️ FRAUD RING: Account belongs to a {graph_score['ring_info']['ring_size']}-member fraud network!")

        # Decision
        if final_prob >= 70:
            risk_level    = 'HIGH'
            action_result = block_transaction(txn_id, account_id, phone,
                              '; '.join(risk_factors), final_prob)
            decision      = 'BLOCKED'
            decision_msg  = 'Transaction BLOCKED. SMS alert sent. Account flagged for review.'
        elif final_prob >= 40:
            risk_level    = 'MEDIUM'
            action_result = flag_for_review(txn_id, account_id, phone,
                              '; '.join(risk_factors), final_prob)
            decision      = 'OTP_REQUIRED'
            decision_msg  = 'OTP verification required. OTP sent to mobile.'
        else:
            risk_level    = 'LOW'
            action_result = approve_transaction(txn_id, phone, final_prob)
            decision      = 'APPROVED'
            decision_msg  = 'Transaction approved. Confirmation SMS sent.'

        TRANSACTION_LOG.append({
            'txn_id': txn_id, 'account_id': account_id,
            'amount': float(data['amount']),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'probability': final_prob, 'risk_level': risk_level,
            'decision': decision, 'risk_factors': risk_factors,
            'sms_id': action_result.get('sms',{}).get('id','N/A')
        })

        return jsonify({
            'txn_id': txn_id, 'account_id': account_id,
            'is_fraud': final_prob >= 50,
            'probability': final_prob,
            'ml_probability': prob_pct,
            'graph_risk': graph_risk,
            'risk_level': risk_level,
            'decision': decision, 'decision_msg': decision_msg,
            'risk_factors': risk_factors,
            'action': action_result,
            'sms_preview': action_result.get('sms',{}).get('message',''),
            'model_used': model_name,
            'model_scores': model_scores,
            'shap': shap_result,
            'graph_analysis': graph_score,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/dashboard')
def dashboard():
    stats = {
        'total':    len(TRANSACTION_LOG),
        'blocked':  sum(1 for t in TRANSACTION_LOG if t['decision']=='BLOCKED'),
        'otp':      sum(1 for t in TRANSACTION_LOG if t['decision']=='OTP_REQUIRED'),
        'approved': sum(1 for t in TRANSACTION_LOG if t['decision']=='APPROVED'),
        'sms_sent': len(SMS_LOG)
    }
    graph_summary = get_graph_summary()
    return render_template('dashboard.html',
        transactions=list(reversed(TRANSACTION_LOG)),
        sms_log=list(reversed(SMS_LOG)),
        blocked=BLOCKED_ACCOUNTS, flagged=FLAGGED_ACCOUNTS,
        stats=stats, graph_summary=graph_summary)

# Enhancement 12: Streaming endpoints
@app.route('/streaming/start', methods=['POST'])
def stream_start():
    started = start_streaming()
    return jsonify({'started': started, 'message': 'Stream started' if started else 'Already running'})

@app.route('/streaming/stop', methods=['POST'])
def stream_stop():
    stop_streaming()
    return jsonify({'stopped': True})

@app.route('/streaming/metrics')
def stream_metrics():
    return jsonify(get_stream_metrics())

# Enhancement 14: Graph endpoints
@app.route('/graph/summary')
def graph_summary_api():
    return jsonify(get_graph_summary())

@app.route('/graph/account/<account_id>')
def graph_account(account_id):
    return jsonify(get_account_graph_score(account_id))

@app.route('/api/log')
def api_log():
    return jsonify(TRANSACTION_LOG)

if __name__ == '__main__':
    print("\n"+"="*60)
    print("  UPI Fraud Detection — All Enhancements Active")
    print("  Enhancement 10: LSTM Ensemble Model")
    print("  Enhancement 11: SHAP Explainability")
    print("  Enhancement 12: Real-Time Streaming")
    print("  Enhancement 13: Mobile REST API")
    print("  Enhancement 14: GNN Fraud Ring Detection")
    print("  App      : http://localhost:5000")
    print("  Dashboard: http://localhost:5000/dashboard")
    print("  Mobile   : http://localhost:5000/mobile/api/v1/health")
    print("="*60+"\n")
    app.run(debug=True, port=5000)
