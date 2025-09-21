import numpy as np
from typeguard import typechecked
import time

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
        # Add bias term (intercept) to X
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Calculate weights using normal equation: w = (X^T * X)^(-1) * X^T * y
        try:
            # Use pseudo-inverse for numerical stability
            self.weights = np.linalg.pinv(X_with_bias) @ y
        except np.linalg.LinAlgError:
            # Fallback to regular inverse if pseudo-inverse fails
            self.weights = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    
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
            raise ValueError("Model must be fitted before making predictions")
        
        # Add bias term to X
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Make predictions: y_pred = X * w
        return X_with_bias @ self.weights

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
    if order < 1:
        raise ValueError("Order must be at least 1")
    
    # Start with the original features
    polynomial_features = [x]
    
    # Add polynomial features up to the specified order
    for degree in range(2, order + 1):
        # Add all combinations of features raised to the current degree
        for i in range(x.shape[1]):
            polynomial_features.append(x[:, i:i+1] ** degree)
        
        # Add cross-product terms for degree 2 and higher
        if degree == 2:
            for i in range(x.shape[1]):
                for j in range(i + 1, x.shape[1]):
                    polynomial_features.append((x[:, i:i+1] * x[:, j:j+1]))
        elif degree == 3:
            for i in range(x.shape[1]):
                for j in range(i + 1, x.shape[1]):
                    polynomial_features.append((x[:, i:i+1] ** 2) * x[:, j:j+1])
                    polynomial_features.append(x[:, i:i+1] * (x[:, j:j+1] ** 2))
        elif degree == 4:
            for i in range(x.shape[1]):
                for j in range(i + 1, x.shape[1]):
                    polynomial_features.append((x[:, i:i+1] ** 3) * x[:, j:j+1])
                    polynomial_features.append((x[:, i:i+1] ** 2) * (x[:, j:j+1] ** 2))
                    polynomial_features.append(x[:, i:i+1] * (x[:, j:j+1] ** 3))
        elif degree == 5:
            for i in range(x.shape[1]):
                for j in range(i + 1, x.shape[1]):
                    polynomial_features.append((x[:, i:i+1] ** 4) * x[:, j:j+1])
                    polynomial_features.append((x[:, i:i+1] ** 3) * (x[:, j:j+1] ** 2))
                    polynomial_features.append((x[:, i:i+1] ** 2) * (x[:, j:j+1] ** 3))
                    polynomial_features.append(x[:, i:i+1] * (x[:, j:j+1] ** 4))
    
    return np.column_stack(polynomial_features)

@typechecked
def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error (RMSE).
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        float: RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

@typechecked
def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        float: R² value
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)

@typechecked
def evaluate_least_squares(X_train: np.ndarray, X_test: np.ndarray, 
                          y_train: np.ndarray, y_test: np.ndarray, 
                          max_order: int = 5) -> dict:
    """
    Evaluate Least Squares models for polynomial orders 1 through max_order.
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Testing features
        y_train (np.ndarray): Training targets
        y_test (np.ndarray): Testing targets
        max_order (int): Maximum polynomial order to evaluate
        
    Returns:
        dict: Results containing RMSE, R², and training times for each order
    """
    results = {}
    
    for order in range(1, max_order + 1):
        print(f"Evaluating polynomial order {order}...")
        
        # Generate polynomial features
        X_train_poly = add_polynomial_features(X_train, order)
        X_test_poly = add_polynomial_features(X_test, order)
        
        # Create and train model
        model = LeastSquares()
        
        # Measure training time
        start_time = time.time()
        model.fit(X_train_poly, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)
        
        # Calculate metrics
        train_rmse = calculate_rmse(y_train, y_train_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)
        train_r2 = calculate_r2(y_train, y_train_pred)
        test_r2 = calculate_r2(y_test, y_test_pred)
        
        results[order] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'training_time': training_time
        }
        
        print(f"Order {order}: Train RMSE={train_rmse:.6f}, Test RMSE={test_rmse:.6f}, "
              f"Train R²={train_r2:.6f}, Test R²={test_r2:.6f}, Time={training_time:.4f}s")
    
    return results