from credit_fraud_test import run_test

if __name__ == '__main__':
    run_test(
        val_path="split/val.csv",
        data_path="split/test.csv",
        model_path="model_xgb_best.pkl"
    )


