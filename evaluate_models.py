import pandas as pd
import numpy as np
import time
import torch as t
from typeguard import typechecked

# Add your own imports here


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
    raise NotImplementedError()
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
    raise NotImplementedError()

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
    raise NotImplementedError()

if __name__ == "__main__":
    run_tests()
