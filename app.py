"""
UPI Fraud Detection - Enhanced Flask App
Features:
  - Transaction blocking
  - SMS alert simulation (Twilio ready)
  - Account flagging & review system
  - Transaction audit log
  - Admin dashboard
Run: python app.py → http://localhost:5000
"""
from flask import Flask, render_template, request, jsonify
import pickle, numpy as np, os, json
from datetime import datetime
import uuid

app = Flask(__name__)

# Load Model
MODEL_PATH = 'models/fraud_model.pkl'
if os.path.exists(MODEL_PATH):
    model    = pickle.load(open('models/fraud_model.pkl', 'rb'))
    scaler   = pickle.load(open('models/scaler.pkl', 'rb'))
    features = pickle.load(open('models/features.pkl', 'rb'))
else:
    model = scaler = features = None

results_data = None
if os.path.exists('models/all_results.pkl'):
    raw = pickle.load(open('models/all_results.pkl', 'rb'))
    results_data = {
        name: {
            'accuracy': round(v['accuracy']*100, 2),
            'f1':       round(v['f1']*100, 2),
            'auc':      round(v['auc']*100, 2),
            'cm':       v['cm'].tolist()
        } for name, v in raw.items()
    }

# In-Memory Storage
TRANSACTION_LOG  = []
BLOCKED_ACCOUNTS = {}
FLAGGED_ACCOUNTS = {}
SMS_LOG          = []

def send_sms(phone, message, txn_id, msg_type):
    sms = {
        'id':        str(uuid.uuid4())[:8].upper(),
        'txn_id':    txn_id,
        'phone':     phone,
        'message':   message,
        'type':      msg_type,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status':    'SENT (Simulated)'
    }
    SMS_LOG.append(sms)
    print(f"\n SMS SENT to {phone}: {message}\n")
    return sms

def block_transaction(txn_id, account_id, phone, reason, probability):
    BLOCKED_ACCOUNTS[account_id] = {
        'txn_id': txn_id, 'reason': reason,
        'blocked_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'probability': probability
    }
    msg = (f"ALERT: Your UPI transaction {txn_id} has been BLOCKED "
           f"due to suspicious activity ({probability:.0f}% fraud risk). "
           f"If this was you, call 1800-XXX-XXXX to unblock. -YourBank")
    sms = send_sms(phone, msg, txn_id, 'FRAUD_ALERT')
    return {'action': 'BLOCKED', 'txn_id': txn_id,
            'message': f'Transaction BLOCKED. SMS alert sent to {phone}.', 'sms': sms}

def flag_for_review(txn_id, account_id, phone, reason, probability):
    otp = str(np.random.randint(100000, 999999))
    FLAGGED_ACCOUNTS[account_id] = {
        'txn_id': txn_id, 'reason': reason,
        'flagged_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'probability': probability, 'status': 'PENDING REVIEW', 'otp': otp
    }
    msg = (f"Your UPI transaction {txn_id} needs verification. "
           f"OTP: {otp} (valid 5 min). Do NOT share. -YourBank")
    sms = send_sms(phone, msg, txn_id, 'OTP_REQUEST')
    return {'action': 'OTP_REQUIRED', 'txn_id': txn_id,
            'message': f'OTP sent to {phone}. Verification required.', 'otp': otp, 'sms': sms}

def approve_transaction(txn_id, phone, probability):
    msg = (f"SUCCESS: UPI Transaction {txn_id} approved. "
           f"Risk Score: {probability:.0f}% (LOW). -YourBank")
    sms = send_sms(phone, msg, txn_id, 'APPROVAL')
    return {'action': 'APPROVED', 'txn_id': txn_id,
            'message': 'Transaction APPROVED. Confirmation SMS sent.', 'sms': sms}

@app.route('/')
def index():
    return render_template('index.html', results=results_data)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded. Run train_model.py first!'}), 500
    try:
        data       = request.json
        account_id = data.get('account_id', 'ACC' + str(np.random.randint(10000,99999)))
        phone      = data.get('phone', '+91-98XXXXXXXX')
        txn_id     = 'TXN' + str(uuid.uuid4())[:8].upper()

        input_data = np.array([[
            float(data['amount']), int(data['hour']),
            int(data['transaction_frequency']), float(data['distance_from_home']),
            int(data['failed_attempts']), int(data['new_device']),
            int(data['vpn_used']), int(data['beneficiary_age_days']),
            int(data['same_bank']), int(data['weekend'])
        ]])

        model_type = type(model).__name__
        if model_type == 'LogisticRegression':
            inp = scaler.transform(input_data)
            prediction  = model.predict(inp)[0]
            probability = model.predict_proba(inp)[0][1]
        else:
            prediction  = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

        prob_pct = round(float(probability) * 100, 2)

        risk_factors = []
        if float(data['amount']) > 10000:        risk_factors.append("High transaction amount (>Rs.10,000)")
        if int(data['hour']) < 6:                risk_factors.append("Unusual time - midnight hours (12AM-5AM)")
        if int(data['transaction_frequency'])>15: risk_factors.append("Abnormally high transaction frequency")
        if float(data['distance_from_home'])>200: risk_factors.append("Transaction far from registered address")
        if int(data['failed_attempts']) >= 3:    risk_factors.append("Multiple failed OTP/PIN attempts")
        if int(data['new_device']) == 1:         risk_factors.append("Transaction from new/unrecognized device")
        if int(data['vpn_used']) == 1:           risk_factors.append("VPN/Proxy connection detected")
        if int(data['beneficiary_age_days']) < 5: risk_factors.append("Beneficiary account very new (<5 days)")

        if prob_pct >= 70:
            risk_level    = 'HIGH'
            action_result = block_transaction(txn_id, account_id, phone,
                              '; '.join(risk_factors) or 'Multiple fraud indicators', prob_pct)
            decision      = 'BLOCKED'
            decision_msg  = 'Transaction BLOCKED. SMS alert sent. Account flagged for review.'
        elif prob_pct >= 40:
            risk_level    = 'MEDIUM'
            action_result = flag_for_review(txn_id, account_id, phone,
                              '; '.join(risk_factors) or 'Suspicious patterns', prob_pct)
            decision      = 'OTP_REQUIRED'
            decision_msg  = 'Suspicious activity! OTP verification required. OTP sent to mobile.'
        else:
            risk_level    = 'LOW'
            action_result = approve_transaction(txn_id, phone, prob_pct)
            decision      = 'APPROVED'
            decision_msg  = 'Transaction approved. Confirmation SMS sent.'

        log_entry = {
            'txn_id': txn_id, 'account_id': account_id,
            'amount': float(data['amount']),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'probability': prob_pct, 'risk_level': risk_level,
            'decision': decision, 'risk_factors': risk_factors,
            'sms_id': action_result.get('sms', {}).get('id', 'N/A')
        }
        TRANSACTION_LOG.append(log_entry)

        return jsonify({
            'txn_id': txn_id, 'account_id': account_id,
            'is_fraud': bool(prediction == 1),
            'probability': prob_pct, 'risk_level': risk_level,
            'decision': decision, 'decision_msg': decision_msg,
            'risk_factors': risk_factors,
            'action': action_result,
            'sms_preview': action_result.get('sms', {}).get('message', ''),
            'model_used': model_type
        })
    except Exception as e:
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
    return render_template('dashboard.html',
        transactions=list(reversed(TRANSACTION_LOG)),
        sms_log=list(reversed(SMS_LOG)),
        blocked=BLOCKED_ACCOUNTS,
        flagged=FLAGGED_ACCOUNTS,
        stats=stats)

@app.route('/api/log')
def api_log():
    return jsonify(TRANSACTION_LOG)

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  UPI Fraud Detection - Enhanced System")
    print("  Main App  : http://localhost:5000")
    print("  Dashboard : http://localhost:5000/dashboard")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
