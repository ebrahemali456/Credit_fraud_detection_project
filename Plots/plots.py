import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix

sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

model_path = "/home/ebrahem-ali/PycharmProjects/Credit_fraud_detection_project/model_xgb_best.pkl"
data_path = "../split/test.csv"

with open(model_path, "rb") as f:
    model_info = pickle.load(f)

if isinstance(model_info, dict):
    model = model_info["model"]
    threshold = model_info.get("threshold", 0.950)
else:
    model = model_info
    threshold = 0.5

print(f"✅ Model loaded (threshold = {threshold})")

data = pd.read_csv(data_path)
X = data.drop(columns=["Class"])
y_true = data["Class"]

y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

precision, recall, _ = precision_recall_curve(y_true, y_prob)
pr_auc = auc(recall, precision)

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)


fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Precision–Recall
axes[0].plot(recall, precision, color='#007bff', linewidth=2)
axes[0].set_title(f'Precision–Recall Curve (AUC = {pr_auc:.3f})', fontsize=13)
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].grid(True, linestyle='--', alpha=0.6)

# ROC
axes[1].plot(fpr, tpr, color='#ff7f0e', linewidth=2, label=f'AUC = {roc_auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.6)
axes[1].set_title(f'ROC Curve (AUC = {roc_auc:.3f})', fontsize=13)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc='lower right')
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix", fontsize=13)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

#Histogram
plt.figure(figsize=(8, 5))
sns.histplot(y_prob[y_true == 0], color='green', label='Class 0', kde=True, bins=40, alpha=0.6)
sns.histplot(y_prob[y_true == 1], color='red', label='Class 1', kde=True, bins=40, alpha=0.6)
plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.3f}')
plt.title("Distribution of Predicted Probabilities", fontsize=13)
plt.xlabel("Predicted Probability")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# Feature Importances
if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(15)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
    plt.title("Top 15 Most Important Features", fontsize=13)
    plt.xlabel("Importance")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.show()
