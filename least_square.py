import numpy as np
from typeguard import typechecked
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk

# Add your own imports here

class LeastSquares:
    @typechecked
    def __init__(self) -> None:
        self.weights = None
        
    @typechecked
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the linear model to the training data using the least squares method.

        Args:
            X (np.ndarray): The input features for training.
            y (np.ndarray): The target variable for training.
        """
        
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        X_aug = np.c_[np.ones(X.shape[0]), X]
        self.weights, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        
        # raise NotImplementedError()
    @typechecked
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for new data using the trained linear model.

        Args:
            X (np.ndarray): The input features for prediction.

        Returns:
            np.ndarray: The predicted target values.
        """
        X = np.asarray(X, dtype=float)
        X_aug = np.c_[np.ones(X.shape[0]), X]
        return X_aug @ self.weights
        #raise NotImplementedError()

if __name__ == "__main__":
    ls = LeastSquares()

    df = pd.read_csv("GasProperties.csv")

    # Features: first 4 columns; Target: 5th column (adjust if your dataset differs)
    # shape (n, 4)
    X = df.iloc[:, :4].to_numpy(dtype=float)
    # shape (n,)  <- 1-D
    y = df.iloc[:, 4].to_numpy(dtype=float)

    ls.fit(X, y)
    print("weights:", ls.weights)  # [intercept, w1, w2, w3, w4]
    preds = ls.predict(X)
    k = 5
    for p, t in zip(preds[:k], y[:k]):
        print(f"pred={p:.6f}  true={t:.6f}")