import argparse
import pickle
import pandas as pd
from sklearn.metrics import precision_recall_curve
from torch.utils.jit.log_extract import run_test

from credit_fraud_utils_data import load_data, preprocess_data
from credit_fraud_utils_eval import evaluate_model


def find_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_threshold

from types import SimpleNamespace

def run_test(val_path, data_path, model_path):
    args = SimpleNamespace(
        val_path=val_path,
        data_path=data_path,
        model_path=model_path
    )
    main(args)


def main(args):
    with open(args.model_path, "rb") as f:
        saved_data = pickle.load(f)

    model = saved_data["model"]
    print(f"Model loaded from {args.model_path}")

    df_val = load_data(args.val_path)
    X_val, y_val = preprocess_data(df_val)
    y_prob_val = model.predict_proba(X_val)[:, 1]
    threshold_val = find_best_threshold(y_val, y_prob_val)
    print(f"Best threshold found on VALIDATION = {threshold_val:.3f}")

    df_test = load_data(args.data_path)
    X_test, y_test = preprocess_data(df_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= threshold_val).astype(int)

    metrics_test = evaluate_model(y_test, y_pred_test, y_prob_test)

    print("\n==============================")
    print("TEST RESULTS (using validation threshold)")
    print("==============================")
    for k, v in metrics_test.items():
        print(f"{k:10}: {v:.4f}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Credit Fraud Detection Model")
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    main(args)



