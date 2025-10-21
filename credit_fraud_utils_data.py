import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def load_data(path):
    data = pd.read_csv(path)
    print(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
    return data


def preprocess_data(data):
    if data is None:
        raise ValueError("Data is None!")

    if 'Class' not in data.columns:
        raise ValueError(" 'Class' column not found in data.")

    X = data.drop('Class', axis=1)
    y = data['Class']

    if X.isnull().sum().any():
        print(" Missing values found — filling with median values.")
        X = X.fillna(X.median())

    return X, y


def balance_data(X, y, method="none"):
    print(f"Balancing data using: {method}")

    if method == "none":
        return X, y

    elif method.lower() == "smote":
        X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)

    elif method.lower() == "oversampling" or method.lower() == "oversample":
        X_resampled, y_resampled = RandomOverSampler().fit_resample(X, y)

    elif method.lower() == "undersampling" or method.lower() == "undersample":
        X_resampled, y_resampled = RandomUnderSampler().fit_resample(X,  y)

    else:
        raise ValueError(" Unknown balancing method. Choose from: none, smote, oversampling, undersampling")

    print(f"Data balanced: Before = {len(y)}, After = {len(y_resampled)}")
    print(f"Class distribution after balancing:\n{pd.Series(y_resampled).value_counts()}")

    return X_resampled, y_resampled


def scale_data(X_train, X_val=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled, scaler
    else:
        return X_train_scaled, scaler



