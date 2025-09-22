import pandas as pd
import numpy as np
from typeguard import typechecked
from typing import Optional

# Add your imports here

class GradientDescent:
    @typechecked
    def __init__(self, learning_rate: float=0.001, n_iterations: int=1000, batch_size: int=32) -> None:
        self.weights = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self._n_features = None
        self._mu = None
        self._sigma = None
                
    @typechecked
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the linear model to the training data using mini-batch Gradient Descent.

        Args:
            X (np.ndarray): The input features for training.
            y (np.ndarray): The target variable for training.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)   # 1-D
        n_samples, n_features = X.shape
        self._n_features = n_features

        # standardize features (prevents huge gradients)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0)
        sigma[sigma == 0.0] = 1.0
        self._mu, self._sigma = mu, sigma
        Xs = (X - mu) / sigma

        # bias term on standardized X
        X_aug = np.c_[np.ones(n_samples), Xs]

        
        # Initialize Weights
        self.weights = np.zeros(n_features + 1)
        
        range_num = np.random.default_rng(0)
        
        batch_size = max(1, min(self.batch_size, n_samples))
        try:
                
            for _ in range(self.n_iterations):
                idx = range_num.permutation(n_samples)
                for start in range(0, n_samples, batch_size):
                    b = idx[start:start + batch_size]
                    Xb = X_aug[b]
                    yb = y[b]

                    # grad of mean-squared error: (X^T (Xw - y)) / m
                    err = Xb @ self.weights - yb
                    grad = (Xb.T @ err) / Xb.shape[0]

                    # update
                    self.weights -= self.learning_rate * grad
        except Exception as e:
            raise RuntimeError("Error during fitting the model.") from e    
            
    @typechecked
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for new data using the trained linear model.

        Args:
            X (np.ndarray): The input features for prediction.

        Returns:
            np.ndarray: The predicted target values.
        """
        if self.weights is None:
            raise RuntimeError("Model not fitted. Call fit(X, y) first.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D for predict.")
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {X.shape[1]}.")

        # apply SAME standardization used in fit
        Xs = (X - self._mu) / self._sigma
        X_aug = np.c_[np.ones(X.shape[0]), Xs]
        return X_aug @ self.weights

if __name__ == "__main__":
    gd = GradientDescent()
    data = pd.read_csv("GasProperties.csv")
    X = data.iloc[:, :4].to_numpy(dtype=float)
    y = data.iloc[:, 4].to_numpy(dtype=float)
    gd.fit(X, y)
    print("weights:", gd.weights)
    preds = gd.predict(X)
    k = 5
    for p, t in zip(preds[:k], y[:k]):
        print(f"pred={p:.6f}  true={t:.6f}")