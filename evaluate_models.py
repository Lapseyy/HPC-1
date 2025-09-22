import pandas as pd
import numpy as np
import time
import torch as t
from typeguard import typechecked
from least_square import LeastSquares
from time import perf_counter
from gradient_descent import GradientDescent


@typechecked
def preprocess(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesses the input data by splitting it into training and testing sets
    and scaling the features. Optionally converts data to PyTorch tensors.

    Args:
        X (np.ndarray): The input features as a NumPy array.
        y (np.ndarray): The target variable as a NumPy array.
        to_tensor (bool): If True, convert the processed data to PyTorch tensors.

    Returns:
        tuple: A tuple containing the processed data. The exact contents depend on `to_tensor`:
            - If to_tensor is False: (X_train_scaled, X_test_scaled, y_train, y_test)
            - If to_tensor is True: (X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)
    """
    """Split (80/20) and standardize features using train statistics."""
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    split = int(0.8 * n)
    tr, te = idx[:split], idx[split:]

    X_train, X_test = X[tr], X[te]
    y_train, y_test = y[tr], y[te]

    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0  # avoid div-by-zero

    X_train = (X_train - mu) / sd
    X_test  = (X_test  - mu) / sd
    return X_train, X_test, y_train, y_test

@typechecked
def add_polynomial_features(x: np.ndarray, order: int) -> np.ndarray:
    """
    Adds polynomial features to the input array X up to the specified order.

    Args:
        x (np.ndarray): The input numpy array of features. Each column represents a feature.
        order (int): The maximum degree of the polynomial features to add.

    Returns:
        np.ndarray: A new numpy array with the original features and the added
                    polynomial features
    """
    x = np.asarray(x, dtype=float)
    feats = [x ** p for p in range(1, order + 1)]
    return np.concatenate(feats, axis=1)
   

def run_tests() -> None:
    """
    Executes a comprehensive evaluation of Least Squares, Gradient Descent, and MLP models
    on the 'GasProperties.csv' dataset. It assesses model performance across different
    polynomial degrees for feature engineering.

    The evaluation process includes:
    - Data loading and optional subsampling.
    - Iterating through polynomial degrees (1 to 5) to observe their impact on model performance.
    - Training and evaluating Least Squares, Gradient Descent, and MLP models.
    - Calculating Root Mean Squared Error (RMSE) and R-squared (R^2) for both training
      and testing datasets.
    - Recording and displaying training times for each model.
    """
    df = pd.read_csv("GasProperties.csv")
    X_raw = df.iloc[:, :4].to_numpy(float)   # features
    y     = df.iloc[:, 4].to_numpy(float)    # target (1-D)

    # 2) split & scale once
    X_tr, X_te, y_tr, y_te = preprocess(X_raw, y)

    # 3) evaluate polynomial orders 1..5
    rows = []
    for d in range(1, 6):
        Xtr_d = add_polynomial_features(X_tr, d)
        Xte_d = add_polynomial_features(X_te, d)

        model = LeastSquares()
        t0 = perf_counter()
        model.fit(Xtr_d, y_tr)
        t1 = perf_counter()

        ytr_hat = model.predict(Xtr_d)
        yte_hat = model.predict(Xte_d)

        rmse_tr = np.sqrt(np.mean((y_tr - ytr_hat) ** 2))
        rmse_te = np.sqrt(np.mean((y_te - yte_hat) ** 2))

        # R^2
        def rsquared(y, yhat):
            sse = np.sum((y - yhat) ** 2)
            sst = np.sum((y - y.mean()) ** 2)
            return 1.0 - sse / sst

        rows.append({
            "Polynomial order": f"Order {d}",
            "Training RMSE": rmse_tr,
            "Training R^2":   rsquared(y_tr, ytr_hat),
            "Training time":  (t1 - t0),
            "Testing RMSE":   rmse_te,
            "Testing R^2":    rsquared(y_te, yte_hat),
        })
        
        # Gradient Descent
        gd = GradientDescent(learning_rate=1e-4, n_iterations=5000, batch_size=len(Xtr_d))
        t0 = perf_counter()
        gd.fit(Xtr_d, y_tr)
        t1 = perf_counter()

        ytr_hat = gd.predict(Xtr_d)
        yte_hat = gd.predict(Xte_d)

        rows.append({
            "Model": "GD",
            "Polynomial order": f"Order {d}",
            "Training RMSE": np.sqrt(np.mean((y_tr - ytr_hat) ** 2)),
            "Training R^2":   rsquared(y_tr, ytr_hat),
            "Training time":  (t1 - t0),
            "Testing RMSE":   np.sqrt(np.mean((y_te - yte_hat) ** 2)),
            "Testing R^2":    rsquared(y_te, yte_hat),
})

    # 4) print the table
    # table = pd.DataFrame(rows)
    # # match the sample formatting
    # with pd.option_context("display.float_format", lambda v: f"{v:.6f}"):
    #     print(table.to_string(index=False))
    table = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:.6f}"):
        print(table.sort_values(["Polynomial order", "Model"]).to_string(index=False))


if __name__ == "__main__":
    run_tests()
