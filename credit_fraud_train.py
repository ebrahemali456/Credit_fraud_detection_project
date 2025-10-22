import argparse
import pickle
import pandas as pd
from xgboost import XGBClassifier

from credit_fraud_utils_data import load_data, preprocess_data, balance_data
from credit_fraud_utils_eval import evaluate_model, find_best_threshold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ======================================================
#  Logistic Regression
# ======================================================
def train_logistic_regression(X_train, y_train):
    print("Training Logistic Regression ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler


# ======================================================
#  Random Forest
# ======================================================
def train_random_forest(X_train, y_train, max_depth=5, n_estimators=20):
    print("Training Random Forest ...")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# ======================================================
#  Voting Classifier
# ======================================================
def train_voting_model(log_model, rf_model, X_train, y_train):
    print("Training Voting Classifier ...")
    voting_clf = VotingClassifier(
        estimators=[('lr', log_model), ('rf', rf_model)],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    return voting_clf


# ======================================================
#  XGBoost
# ======================================================
def train_xgboost(X_train, y_train, X_val, y_val,
                  max_depth=8, n_estimators=400, learning_rate=0.05):
    print(" Training XGBoost ...")
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        scale_pos_weight=2,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model


# ======================================================
#  MAIN FUNCTION
# ======================================================
def main(args):
    print(f" Loading data from: {args.data_path}")
    try:
        df = load_data(args.data_path)
    except FileNotFoundError:
        print(f" Error: File not found at {args.data_path}")
        return

    X, y = preprocess_data(df)
    X, y = balance_data(X, y, method=args.balance)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ====================
    # Train Selected Model
    # ====================
    if args.model == "logistic":
        model, scaler = train_logistic_regression(X_train, y_train)
        X_val_scaled = scaler.transform(X_val)
        y_prob = model.predict_proba(X_val_scaled)[:, 1]

    elif args.model == "randomforest":
        model = train_random_forest(X_train, y_train, args.max_depth, args.n_estimators)
        y_prob = model.predict_proba(X_val)[:, 1]

    elif args.model == "xgboost":
        model = train_xgboost(X_train, y_train, X_val, y_val,
                              max_depth=args.max_depth,
                              n_estimators=args.n_estimators)
        y_prob = model.predict_proba(X_val)[:, 1]

    elif args.model == "voting":
        log_model, scaler = train_logistic_regression(X_train, y_train)
        rf_model = train_random_forest(X_train, y_train, args.max_depth, args.n_estimators)
        model = train_voting_model(log_model, rf_model, X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]

    else:
        raise ValueError("Unknown model type. Choose from: logistic, randomforest, xgboost, voting")

    # ====================
    # Evaluation
    # ====================
    print("\n Evaluating model ...")
    best_threshold, metrics = find_best_threshold(y_val, y_prob)
    print(f"\n Best Threshold: {best_threshold:.3f}")
    print(metrics)

    # ====================
    # Save Model
    # ====================
    with open(args.save_path, "wb") as f:
        pickle.dump({"model": model, "best_threshold": best_threshold}, f)

    print(f"\n Model saved successfully to: {args.save_path}")


# ======================================================
#  ENTRY POINT
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Credit Card Fraud Detection Model")

    parser.add_argument("--data_path", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--model", type=str,
                        choices=["logistic", "randomforest", "xgboost", "voting"],
                        required=True)
    parser.add_argument("--save_path", type=str, default="model.pkl")
    parser.add_argument("--balance", type=str,
                        choices=["none", "smote", "oversample", "undersampling"],
                        default="none")
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--n_estimators", type=int, default=400)

    args = parser.parse_args()
    main(args)
