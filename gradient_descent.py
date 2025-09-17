import pandas as pd
import numpy as np
from typeguard import typechecked

# Add your imports here

class GradientDescent:
    @typechecked
    def __init__(self, learning_rate: float=0.001, n_iterations: int=1000, batch_size: int=32) -> None:
        self.weights = None
        raise NotImplementedError()
        
    @typechecked
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the linear model to the training data using mini-batch Gradient Descent.

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
    
