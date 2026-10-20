"""
UPI Fraud Detection - Exploratory Data Analysis
Run: python eda.py
Generates charts saved in data/ folder
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('dark_background')
sns.set_theme(style="darkgrid")

if not os.path.exists('data/upi_transactions.csv'):
    print("Dataset not found. Run generate_data.py first!")
    exit()

df = pd.read_csv('data/upi_transactions.csv')
print(f"Dataset: {len(df)} transactions")

os.makedirs('data/charts', exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('UPI Fraud Detection - EDA', fontsize=16, color='white', y=1.02)

# 1. Fraud Distribution
fraud_counts = df['is_fraud'].value_counts()
axes[0,0].pie(fraud_counts, labels=['Legitimate', 'Fraud'], autopct='%1.1f%%',
              colors=['#22c55e', '#ef4444'], startangle=90)
axes[0,0].set_title('Transaction Distribution')

# 2. Amount Distribution
df[df['is_fraud']==0]['amount'].hist(ax=axes[0,1], bins=50, color='#22c55e', alpha=0.7, label='Legitimate')
df[df['is_fraud']==1]['amount'].hist(ax=axes[0,1], bins=50, color='#ef4444', alpha=0.7, label='Fraud')
axes[0,1].set_title('Transaction Amount Distribution')
axes[0,1].set_xlabel('Amount (₹)')
axes[0,1].legend()

# 3. Hour vs Fraud
fraud_by_hour = df.groupby(['hour','is_fraud']).size().unstack(fill_value=0)
fraud_by_hour.plot(ax=axes[0,2], color=['#22c55e','#ef4444'])
axes[0,2].set_title('Transactions by Hour')
axes[0,2].set_xlabel('Hour of Day')
axes[0,2].legend(['Legitimate','Fraud'])

# 4. Correlation Heatmap
num_cols = ['amount','hour','transaction_frequency','distance_from_home',
            'failed_attempts','new_device','vpn_used','beneficiary_age_days','is_fraud']
corr = df[num_cols].corr()
sns.heatmap(corr, ax=axes[1,0], cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5, annot_kws={'size':7})
axes[1,0].set_title('Feature Correlation')

# 5. Failed Attempts vs Fraud
sns.boxplot(data=df, x='is_fraud', y='failed_attempts', ax=axes[1,1],
            palette={0:'#22c55e', 1:'#ef4444'})
axes[1,1].set_xticklabels(['Legitimate','Fraud'])
axes[1,1].set_title('Failed Attempts vs Fraud')

# 6. Distance from home
sns.boxplot(data=df, x='is_fraud', y='distance_from_home', ax=axes[1,2],
            palette={0:'#22c55e', 1:'#ef4444'})
axes[1,2].set_xticklabels(['Legitimate','Fraud'])
axes[1,2].set_title('Distance from Home vs Fraud')

plt.tight_layout()
plt.savefig('data/charts/eda_report.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print("EDA charts saved to: data/charts/eda_report.png")

# Summary Stats
print("\n--- Summary Statistics ---")
print(df.groupby('is_fraud')[['amount','transaction_frequency','distance_from_home','failed_attempts']].mean().round(2))
