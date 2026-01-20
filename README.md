# Credit Card Fraud Detection (Imbalanced Classification)

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques on a highly imbalanced dataset (fraud rate ≈ 0.17%).  
The main goal is to maximize fraud detection performance while maintaining a strong balance between **precision** and **recall**, with particular emphasis on the **F1-score**.

---

## 📊 Dataset
- **Source:** Credit Card Fraud Detection Dataset (Anonymized Transactions)
- **Total Samples:** 284,807
- **Fraud Cases:** 492 (~0.17%)
- **Features:** 30 numerical features (V1–V28, Amount, Time)
- **Target Column:**  
  - `0` → Non-Fraud  
  - `1` → Fraud

### Data Splitting
| Set        | Samples |
|------------|---------|
| Training   | 170,884 |
| Validation | 56,960  |
| Test       | 56,960  |

### After Applying SMOTE
| Class | Before | After |
|------|--------|-------|
| Non-Fraud (0) | 170,579 | 170,579 |
| Fraud (1)     | 305     | 170,579 |

---

## 🧹 Data Preprocessing
- Removed duplicates and handled missing values
- Feature scaling using **StandardScaler**
- Addressed class imbalance using:
  - SMOTE
  - Oversampling
  - Undersampling
- Split data into training, validation, and test sets

---

## 🤖 Model Training
- **Algorithm:** XGBoost Classifier  
- **Why XGBoost?**
  - Handles imbalanced data effectively
  - Captures non-linear relationships
  - Built-in regularization to prevent overfitting

### Training Configuration
| Parameter        | Value |
|------------------|-------|
| max_depth        | 8     |
| n_estimators     | 400   |
| learning_rate    | 0.05  |
| scale_pos_weight | 2     |

---

## 📈 Model Performance

### Training Results
| Metric    | Value  |
|-----------|--------|
| F1-Score  | 0.9999 |
| Precision | 0.9999 |
| Recall    | 1.0000 |
| PR-AUC    | 0.9999 |

### Validation Results
| Metric    | Value  |
|-----------|--------|
| F1-Score  | 0.8690 |
| Precision | 0.9359 |
| Recall    | 0.8111 |
| PR-AUC    | 0.8537 |

### Test Results (Threshold = 0.950)
| Metric    | Value  |
|-----------|--------|
| F1-Score  | 0.8718 |
| Precision | 0.8673 |
| Recall    | 0.8763 |
| PR-AUC    | 0.8701 |

---

## 📉 Visual Analysis
The following visualizations were used to analyze model performance:
- Precision–Recall Curve
- ROC Curve
- Confusion Matrix
- Predicted Probability Distribution
- Feature Importance Plot

### Top Important Features
- **V14**
- **V12**
- **V10**

---

## 🧠 Conclusion
The XGBoost model achieved strong generalization on unseen data with an **F1-score of 0.87** and **Recall of 0.88** on the test set.  
Its robustness in handling severe class imbalance makes it well-suited for real-world fraud detection systems.

---

## 🛠️ Technologies Used
- **Programming Language:** Python 3.12
- **Libraries:**  
  - XGBoost  
  - Scikit-learn  
  - Pandas  
  - NumPy  
  - Matplotlib  
  - Seaborn

---

## 🚀 How to Run the Project
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
python main.py
