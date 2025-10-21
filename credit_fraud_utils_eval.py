import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, precision_recall_curve, \
    classification_report


def evaluate_model(y_true, y_pred , y_prob = None):
    metrics = {'f1_score' : f1_score(y_true, y_pred),
              'precision' : precision_score(y_true, y_pred),
              'recall' : recall_score(y_true, y_pred)}
    if y_prob is not None:
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)

    return metrics


def find_best_threshold(y_true, y_prob):


    precisions , recallas , thresholds = precision_recall_curve(y_true, y_prob)
    f1_score = 2 * precisions * recallas / (precisions + recallas)
    best_idx = np.argmax(f1_score)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred_best = (y_prob >= best_threshold).astype(int)
    best_metrics = evaluate_model(y_true, y_pred_best, y_prob)

    return best_threshold, best_metrics


def print_classification_summary(y_true, y_pred):

    print("\n" + "="*50)
    print("📊 Classification Report")
    print("="*50)
    print(classification_report(y_true, y_pred, digits=4))


