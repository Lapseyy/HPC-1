import numpy as np
from typeguard import typechecked

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
        raise NotImplementedError()
    @typechecked
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for new data using the trained linear model.

        Args:
            X (np.ndarray): The input features for prediction.

        Returns:
            np.ndarray: The predicted target values.
        """
        raise NotImplementedError()