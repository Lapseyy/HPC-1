import numpy as np
from typeguard import typechecked
import time

# Add your imports here

class GradientDescent:
    @typechecked
    def __init__(self, learning_rate: float=0.001, n_iterations: int=1000, batch_size: int=32) -> None:
        """
        Initialize the Gradient Descent model.
        
        Args:
            learning_rate (float): The learning rate for gradient descent.
            n_iterations (int): Maximum number of iterations.
            batch_size (int): Size of mini-batches for training.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    @typechecked
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the linear model to the training data using mini-batch Gradient Descent.

        Args:
            X (np.ndarray): The input features for training.
            y (np.ndarray): The target variable for training.
        """
        # Initialize weights and bias
        n_features = X.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
        
        # Keep y as 1D for consistency
        if y.ndim > 1:
            y = y.flatten()
        
        # Mini-batch gradient descent
        n_samples = X.shape[0]
        
        for iteration in range(self.n_iterations):
            # Shuffle the data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process mini-batches
            for i in range(0, n_samples, self.batch_size):
                # Get mini-batch
                end_idx = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                # Forward pass
                y_pred = self._predict_batch(X_batch)
                
                # Calculate gradients
                m = X_batch.shape[0]
                error = y_pred - y_batch
                
                # Ensure error is 1D
                if error.ndim > 1:
                    error = error.flatten()
                
                # Gradient of weights: (1/m) * X^T * error
                dw = (1/m) * np.dot(X_batch.T, error)
                
                # Gradient of bias: (1/m) * sum(error)
                db = (1/m) * np.sum(error)
                
                # Ensure dw is 1D
                if dw.ndim > 1:
                    dw = dw.flatten()
                
                # Update parameters
                self.weights = self.weights - self.learning_rate * dw
                self.bias = self.bias - self.learning_rate * db
            
            # Calculate cost for monitoring (optional)
            if iteration % 100 == 0 or iteration == self.n_iterations - 1:
                y_pred_full = self._predict_batch(X)
                cost = self._calculate_cost(y, y_pred_full)
                self.cost_history.append(cost)
    
    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Internal method to make predictions for a batch of data.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predictions
        """
        return (X @ self.weights + self.bias).flatten()
    
    def _calculate_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate mean squared error cost.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            float: Mean squared error
        """
        return np.mean((y_true - y_pred) ** 2)
    
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
        
        return self._predict_batch(X)

@typechecked
def add_polynomial_features(x: np.ndarray, order: int) -> np.ndarray:
    """
    Adds polynomial features to the input array X up to the specified order.
    This is the same implementation as in least_square.py for consistency.

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
def evaluate_gradient_descent(X_train: np.ndarray, X_test: np.ndarray, 
                             y_train: np.ndarray, y_test: np.ndarray, 
                             max_order: int = 5, learning_rate: float = 0.001, 
                             n_iterations: int = 1000, batch_size: int = 32) -> dict:
    """
    Evaluate Gradient Descent models for polynomial orders 1 through max_order.
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Testing features
        y_train (np.ndarray): Training targets
        y_test (np.ndarray): Testing targets
        max_order (int): Maximum polynomial order to evaluate
        learning_rate (float): Learning rate for gradient descent
        n_iterations (int): Number of iterations
        batch_size (int): Batch size for mini-batch gradient descent
        
    Returns:
        dict: Results containing RMSE, R², and training times for each order
    """
    results = {}
    
    for order in range(1, max_order + 1):
        print(f"Evaluating polynomial order {order} with Gradient Descent...")
        
        # Generate polynomial features
        X_train_poly = add_polynomial_features(X_train, order)
        X_test_poly = add_polynomial_features(X_test, order)
        
        # Create and train model
        model = GradientDescent(learning_rate=learning_rate, 
                               n_iterations=n_iterations, 
                               batch_size=batch_size)
        
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
            'training_time': training_time,
            'cost_history': model.cost_history
        }
        
        print(f"Order {order}: Train RMSE={train_rmse:.6f}, Test RMSE={test_rmse:.6f}, "
              f"Train R²={train_r2:.6f}, Test R²={test_r2:.6f}, Time={training_time:.4f}s")
    
    return results