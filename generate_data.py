"""
UPI Fraud Transaction Dataset Generator
Creates realistic overlapping distributions → LOW / MEDIUM / HIGH risk zones
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
N_TOTAL   = 12000
FRAUD_RATE = 0.08

n_fraud = int(N_TOTAL * FRAUD_RATE)
n_legit = N_TOTAL - n_fraud

def gen_legit(n):
    return pd.DataFrame({
        'amount':                np.random.exponential(2000, n).clip(50, 9000).round(2),
        'hour':                  np.random.randint(8, 22, n),
        'transaction_frequency': np.random.randint(1, 8, n),
        'distance_from_home':    np.random.exponential(15, n).clip(0, 80).round(2),
        'failed_attempts':       np.random.choice([0,0,0,0,1,1,2], n),
        'new_device':            np.random.choice([0,0,0,0,0,1], n),
        'vpn_used':              np.random.choice([0,0,0,0,0,1], n),
        'beneficiary_age_days':  np.random.randint(30, 1000, n),
        'same_bank':             np.random.choice([1,1,1,0], n),
        'weekend':               np.random.choice([0,0,1], n),
        'is_fraud':              0
    })

def gen_fraud_high(n):
    amt = np.where(np.random.rand(n) > 0.3,
                   np.random.uniform(25000,50000,n),
                   np.random.uniform(10,80,n))
    return pd.DataFrame({
        'amount':                amt.round(2),
        'hour':                  np.random.randint(0, 5, n),
        'transaction_frequency': np.random.randint(20, 60, n),
        'distance_from_home':    np.random.randint(500, 5000, n),
        'failed_attempts':       np.random.randint(3, 8, n),
        'new_device':            1,
        'vpn_used':              1,
        'beneficiary_age_days':  np.random.randint(0, 4, n),
        'same_bank':             0,
        'weekend':               np.random.choice([0,1], n),
        'is_fraud':              1
    })

def gen_fraud_medium(n):
    return pd.DataFrame({
        'amount':                np.random.randint(8000, 25000, n).astype(float),
        'hour':                  np.random.choice([5,6,7,8,20,21,22,23], n),
        'transaction_frequency': np.random.randint(10, 22, n),
        'distance_from_home':    np.random.randint(100, 600, n).astype(float),
        'failed_attempts':       np.random.choice([1,2,3], n),
        'new_device':            np.random.choice([0,1,1], n),
        'vpn_used':              np.random.choice([0,1,1], n),
        'beneficiary_age_days':  np.random.randint(3, 20, n),
        'same_bank':             np.random.choice([0,1], n),
        'weekend':               np.random.choice([0,1], n),
        'is_fraud':              1
    })

def gen_fraud_low(n):
    return pd.DataFrame({
        'amount':                np.random.randint(3000, 12000, n).astype(float),
        'hour':                  np.random.choice([6,7,8,9,19,20,21,22], n),
        'transaction_frequency': np.random.randint(8, 15, n),
        'distance_from_home':    np.random.randint(50, 200, n).astype(float),
        'failed_attempts':       np.random.choice([1,2], n),
        'new_device':            np.random.choice([0,1], n),
        'vpn_used':              np.random.choice([0,0,1], n),
        'beneficiary_age_days':  np.random.randint(5, 30, n),
        'same_bank':             np.random.choice([0,1], n),
        'weekend':               np.random.choice([0,1], n),
        'is_fraud':              1
    })

n_high = int(n_fraud * 0.50)
n_med  = int(n_fraud * 0.35)
n_low  = n_fraud - n_high - n_med

legit  = gen_legit(n_legit)
high   = gen_fraud_high(n_high)
medium = gen_fraud_medium(n_med)
low    = gen_fraud_low(n_low)

df = pd.concat([legit, high, medium, low], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

int_cols = ['hour','transaction_frequency','failed_attempts','new_device',
            'vpn_used','same_bank','weekend','is_fraud','beneficiary_age_days']
for c in int_cols:
    df[c] = df[c].astype(int)
df['transaction_id'] = ['TXN' + str(i).zfill(6) for i in range(len(df))]

os.makedirs('data', exist_ok=True)
df.to_csv('data/upi_transactions.csv', index=False)
print(f"Dataset: {len(df)} transactions")
print(df['is_fraud'].value_counts().rename({0:'Legitimate',1:'Fraud'}))
print(f"Fraud rate: {df['is_fraud'].mean()*100:.1f}%")
