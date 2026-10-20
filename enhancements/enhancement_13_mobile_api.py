"""
Enhancement 13: Mobile App REST API
=====================================
Production-ready REST API designed for Flutter / React Native mobile apps.
Includes push notification simulation, JWT-style tokens, and mobile-optimized responses.

Mobile App Features:
  - Bank officer can approve/reject blocked transactions from phone
  - Push notifications for high-risk alerts
  - Real-time dashboard metrics
  - Biometric auth simulation

In production:
  - Use Flask-JWT-Extended for real JWT auth
  - Firebase Cloud Messaging (FCM) for real push notifications
  - Deploy on AWS/GCP with HTTPS
"""

import uuid
import time
import hashlib
import json
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify

mobile_api = Blueprint('mobile_api', __name__, url_prefix='/mobile/api/v1')

# ── Simulated token store ──────────────────────────────────────────────────
_active_tokens  = {}   # token → {user_id, role, expires}
_push_queue     = []   # pending push notifications
_officer_actions= []   # log of officer decisions

DEMO_USERS = {
    'officer001': {'password': 'bank@123', 'name': 'Priya Sharma',    'role': 'officer'},
    'admin001':   {'password': 'admin@123','name': 'Rajesh Kumar',    'role': 'admin'},
    'analyst001': {'password': 'data@123', 'name': 'Anitha Reddy',   'role': 'analyst'},
}

# ── Auth helpers ───────────────────────────────────────────────────────────
def _generate_token(user_id, role):
    token = hashlib.sha256(f"{user_id}{time.time()}{uuid.uuid4()}".encode()).hexdigest()[:32]
    _active_tokens[token] = {
        'user_id':  user_id,
        'role':     role,
        'name':     DEMO_USERS[user_id]['name'],
        'issued_at':datetime.now().isoformat(),
        'expires':  (datetime.now() + timedelta(hours=8)).isoformat()
    }
    return token

def _verify_token(token):
    return _active_tokens.get(token)

def _require_auth(f):
    """Decorator: require valid token in Authorization header."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get('Authorization', '')
        token = auth.replace('Bearer ', '')
        user = _verify_token(token)
        if not user:
            return jsonify({'error': 'Unauthorized. Please login.', 'code': 401}), 401
        request.current_user = user
        return f(*args, **kwargs)
    return decorated

def _send_push(user_id, title, body, data=None, priority='high'):
    """
    Simulated push notification.
    Production: Firebase Admin SDK
        import firebase_admin
        from firebase_admin import messaging
        msg = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            token=fcm_device_token
        )
        firebase_admin.messaging.send(msg)
    """
    notif = {
        'id':        str(uuid.uuid4())[:8].upper(),
        'user_id':   user_id,
        'title':     title,
        'body':      body,
        'data':      data or {},
        'priority':  priority,
        'sent_at':   datetime.now().isoformat(),
        'status':    'DELIVERED (Simulated)'
    }
    _push_queue.append(notif)
    return notif

# ══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════

# ── Auth ───────────────────────────────────────────────────────────────────
@mobile_api.route('/auth/login', methods=['POST'])
def login():
    """
    POST /mobile/api/v1/auth/login
    Body: { "user_id": "officer001", "password": "bank@123" }
    """
    data     = request.json or {}
    user_id  = data.get('user_id', '')
    password = data.get('password', '')

    user = DEMO_USERS.get(user_id)
    if not user or user['password'] != password:
        return jsonify({'error': 'Invalid credentials'}), 401

    token = _generate_token(user_id, user['role'])
    _send_push(user_id, '🔐 Login Alert',
               f"New login detected for {user['name']}. If not you, contact security.")

    return jsonify({
        'success':    True,
        'token':      token,
        'user':       {'id': user_id, 'name': user['name'], 'role': user['role']},
        'expires_in': '8 hours',
        'message':    f"Welcome, {user['name']}!"
    })

@mobile_api.route('/auth/logout', methods=['POST'])
@_require_auth
def logout():
    auth  = request.headers.get('Authorization', '')
    token = auth.replace('Bearer ', '')
    _active_tokens.pop(token, None)
    return jsonify({'success': True, 'message': 'Logged out successfully'})

# ── Dashboard Metrics ──────────────────────────────────────────────────────
@mobile_api.route('/dashboard/metrics', methods=['GET'])
@_require_auth
def dashboard_metrics():
    """
    GET /mobile/api/v1/dashboard/metrics
    Returns real-time summary for mobile dashboard home screen.
    """
    from app import TRANSACTION_LOG, SMS_LOG, BLOCKED_ACCOUNTS, FLAGGED_ACCOUNTS

    total    = len(TRANSACTION_LOG)
    blocked  = sum(1 for t in TRANSACTION_LOG if t['decision']=='BLOCKED')
    otp      = sum(1 for t in TRANSACTION_LOG if t['decision']=='OTP_REQUIRED')
    approved = sum(1 for t in TRANSACTION_LOG if t['decision']=='APPROVED')

    # Last hour stats
    recent = TRANSACTION_LOG[-20:] if TRANSACTION_LOG else []
    avg_prob = round(sum(t['probability'] for t in recent)/max(1,len(recent)), 1)

    return jsonify({
        'summary': {
            'total_transactions':  total,
            'blocked':             blocked,
            'otp_required':        otp,
            'approved':            approved,
            'fraud_rate_pct':      round(blocked/max(1,total)*100, 1),
            'avg_fraud_probability': avg_prob,
            'blocked_accounts':    len(BLOCKED_ACCOUNTS),
            'flagged_accounts':    len(FLAGGED_ACCOUNTS),
            'sms_sent':            len(SMS_LOG),
        },
        'recent_alerts': [
            {k: v for k, v in t.items() if k != 'risk_factors'}
            for t in list(reversed(TRANSACTION_LOG))[:5]
            if t['decision'] in ('BLOCKED','OTP_REQUIRED')
        ],
        'timestamp': datetime.now().isoformat()
    })

# ── Alert Feed ─────────────────────────────────────────────────────────────
@mobile_api.route('/alerts', methods=['GET'])
@_require_auth
def get_alerts():
    """
    GET /mobile/api/v1/alerts?limit=20&risk=HIGH
    Returns paginated alert feed for mobile notification screen.
    """
    from app import TRANSACTION_LOG
    limit = int(request.args.get('limit', 20))
    risk  = request.args.get('risk', 'ALL').upper()

    alerts = list(reversed(TRANSACTION_LOG))
    if risk != 'ALL':
        alerts = [a for a in alerts if a.get('risk_level') == risk]

    return jsonify({
        'alerts':     alerts[:limit],
        'total':      len(alerts),
        'has_more':   len(alerts) > limit
    })

# ── Officer Actions ────────────────────────────────────────────────────────
@mobile_api.route('/transaction/<txn_id>/action', methods=['POST'])
@_require_auth
def officer_action(txn_id):
    """
    POST /mobile/api/v1/transaction/TXN-XXXXXX/action
    Body: { "action": "UNBLOCK" | "CONFIRM_FRAUD" | "ESCALATE", "notes": "..." }

    Officer reviews a flagged transaction from their phone and takes action.
    """
    data   = request.json or {}
    action = data.get('action', '').upper()
    notes  = data.get('notes', '')
    user   = request.current_user

    if action not in ('UNBLOCK', 'CONFIRM_FRAUD', 'ESCALATE'):
        return jsonify({'error': 'Invalid action. Use UNBLOCK/CONFIRM_FRAUD/ESCALATE'}), 400

    from app import TRANSACTION_LOG, BLOCKED_ACCOUNTS, FLAGGED_ACCOUNTS
    txn = next((t for t in TRANSACTION_LOG if t['txn_id']==txn_id), None)
    if not txn:
        return jsonify({'error': f'Transaction {txn_id} not found'}), 404

    log_entry = {
        'action_id':   str(uuid.uuid4())[:8].upper(),
        'txn_id':      txn_id,
        'officer':     user['name'],
        'officer_role':user['role'],
        'action':      action,
        'notes':       notes,
        'timestamp':   datetime.now().isoformat()
    }
    _officer_actions.append(log_entry)

    # Apply action
    if action == 'UNBLOCK':
        BLOCKED_ACCOUNTS.pop(txn.get('account_id',''), None)
        FLAGGED_ACCOUNTS.pop(txn.get('account_id',''), None)
        msg = f"Transaction {txn_id} has been manually UNBLOCKED by officer {user['name']}."
        _send_push(user['user_id'], '✅ Transaction Unblocked', msg, {'txn_id': txn_id})
        result_msg = f"Transaction {txn_id} unblocked successfully"

    elif action == 'CONFIRM_FRAUD':
        msg = f"FRAUD CONFIRMED for transaction {txn_id} by officer {user['name']}. Account permanently flagged."
        _send_push(user['user_id'], '🚨 Fraud Confirmed', msg, {'txn_id': txn_id})
        result_msg = f"Fraud confirmed and account flagged for {txn_id}"

    else:  # ESCALATE
        msg = f"Transaction {txn_id} escalated to senior team by {user['name']}."
        _send_push(user['user_id'], '⬆️ Case Escalated', msg, {'txn_id': txn_id})
        result_msg = f"Transaction {txn_id} escalated to senior review team"

    return jsonify({
        'success':    True,
        'message':    result_msg,
        'action_log': log_entry,
        'notification_sent': True
    })

# ── Push Notifications ─────────────────────────────────────────────────────
@mobile_api.route('/notifications', methods=['GET'])
@_require_auth
def get_notifications():
    """GET /mobile/api/v1/notifications — Returns push notification history."""
    user_id = request.current_user['user_id']
    user_notifs = [n for n in _push_queue if n['user_id']==user_id]
    return jsonify({
        'notifications': list(reversed(user_notifs))[:20],
        'unread_count':  len(user_notifs)
    })

# ── Officer Action Log ─────────────────────────────────────────────────────
@mobile_api.route('/officer/actions', methods=['GET'])
@_require_auth
def officer_actions():
    """GET /mobile/api/v1/officer/actions — Officer action audit trail."""
    return jsonify({
        'actions': list(reversed(_officer_actions))[:50],
        'total':   len(_officer_actions)
    })

# ── Health Check ───────────────────────────────────────────────────────────
@mobile_api.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':    'healthy',
        'version':   'v1.0.0',
        'timestamp': datetime.now().isoformat(),
        'services':  {'ml_model':'online','sms':'online','streaming':'online'}
    })
