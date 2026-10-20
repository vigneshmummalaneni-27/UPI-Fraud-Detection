# 🛡️ UPI Fraud Detection System
### Final Year Project | Machine Learning | B.E./B.Tech CSE

---

## 📋 Project Overview
A machine learning-based system that detects fraudulent UPI transactions in real-time using
Random Forest, Gradient Boosting, and Logistic Regression with SMOTE balancing.

---

## 🚀 QUICK SETUP (Step by Step)

### Step 1: Install Python
Download Python 3.10+ from https://python.org and install it.

### Step 2: Open Terminal / Command Prompt
Navigate to this project folder:
```
cd upi_fraud_detection
```

### Step 3: Install Required Libraries
```
pip install -r requirements.txt
```

### Step 4: Generate Dataset
```
python generate_data.py
```

### Step 5: Train the ML Model
```
python train_model.py
```

### Step 6: Run the Web Application
```
python app.py
```

### Step 7: Open in Browser
Go to: **http://localhost:5000**

---

## 📁 Project Structure

```
upi_fraud_detection/
│
├── generate_data.py      ← Step 1: Creates synthetic UPI dataset
├── train_model.py        ← Step 2: Trains ML models
├── app.py                ← Step 3: Flask web application
├── eda.py                ← Optional: Generates EDA charts
├── requirements.txt      ← Python dependencies
├── README.md             ← This file
│
├── data/
│   └── upi_transactions.csv   ← Generated dataset
│
├── models/
│   ├── fraud_model.pkl         ← Trained best model
│   ├── scaler.pkl              ← Feature scaler
│   └── all_results.pkl         ← All model results
│
└── templates/
    └── index.html              ← Web interface
```

---

## 🤖 ML Models Used

| Model               | Purpose                        |
|---------------------|--------------------------------|
| Random Forest        | Main detection model           |
| Gradient Boosting    | High accuracy ensemble method  |
| Logistic Regression  | Baseline model                 |
| SMOTE                | Handle class imbalance         |

---

## 📊 Features Used for Detection

| Feature               | Description                          |
|-----------------------|--------------------------------------|
| amount                | Transaction amount in ₹              |
| hour                  | Hour of day (0-23)                   |
| transaction_frequency | Number of transactions today         |
| distance_from_home    | Distance from registered location    |
| failed_attempts       | Failed login/OTP attempts            |
| new_device            | Transaction from new device          |
| vpn_used              | VPN detected                         |
| beneficiary_age_days  | How old is the beneficiary account   |
| same_bank             | Same bank or inter-bank transfer     |
| weekend               | Weekend or weekday transaction       |

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **ML Libraries:** Scikit-learn, Imbalanced-learn (SMOTE)
- **Web Framework:** Flask
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** HTML, CSS, JavaScript

---

## 📈 Expected Results

- Accuracy: ~95-98%
- F1-Score: ~90-95%
- ROC-AUC: ~95-99%

---

## 👨‍💻 Developed For
**Final Year Engineering Project**
Department of Computer Science & Engineering

---
*Note: This project uses synthetic data for demonstration purposes.*
