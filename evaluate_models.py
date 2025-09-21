import pandas as pd
import numpy as np
import time
import torch as t
from typeguard import typechecked

# Add your own imports here
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from least_square import LeastSquares, add_polynomial_features as ls_add_poly, calculate_rmse, calculate_r2, evaluate_least_squares
from gradient_descent import GradientDescent, add_polynomial_features as gd_add_poly, evaluate_gradient_descent
from mlp import MLP, setup_model, train_model


@typechecked
def preprocess(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesses the input data by splitting it into training and testing sets
    and scaling the features. Optionally converts data to PyTorch tensors.

    Args:
        X (np.ndarray): The input features as a NumPy array.
        y (np.ndarray): The target variable as a NumPy array.

    Returns:
        tuple: A tuple containing the processed data:
            - (X_train_scaled, X_test_scaled, y_train, y_test)
    """
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
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
    # Use the same implementation as in least_square.py for consistency
    return ls_add_poly(x, order)

@typechecked
def evaluate_mlp(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, order: int) -> dict:
    """
    Evaluate MLP model for a specific polynomial order.
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Testing features  
        y_train (np.ndarray): Training targets
        y_test (np.ndarray): Testing targets
        order (int): Polynomial order
        
    Returns:
        dict: Results containing RMSE, R², and training time
    """
    try:
        # Convert to PyTorch tensors
        X_train_tensor = t.FloatTensor(X_train)
        X_test_tensor = t.FloatTensor(X_test)
        y_train_tensor = t.FloatTensor(y_train.reshape(-1, 1))
        y_test_tensor = t.FloatTensor(y_test.reshape(-1, 1))
        
        # Create data loader
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Setup model - use a reasonable architecture based on input size
        input_size = X_train.shape[1]
        hidden_layers = [min(64, input_size * 4), min(32, input_size * 2)]  # Adaptive architecture
        output_size = 1
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        
        model, criterion, optimizer = setup_model(input_size, hidden_layers, output_size, device)
        
        # Move data to device
        X_train_tensor = X_train_tensor.to(device)
        X_test_tensor = X_test_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)
        
        # Move data loader to device
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train model
        start_time = time.time()
        trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=100)
        training_time = time.time() - start_time
        
        # Make predictions
        trained_model.eval()
        with t.no_grad():
            y_train_pred = trained_model(X_train_tensor).cpu().numpy().flatten()
            y_test_pred = trained_model(X_test_tensor).cpu().numpy().flatten()
        
        # Calculate metrics
        train_rmse = calculate_rmse(y_train, y_train_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)
        train_r2 = calculate_r2(y_train, y_train_pred)
        test_r2 = calculate_r2(y_test, y_test_pred)
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'training_time': training_time
        }
        
    except Exception as e:
        print(f"    Error in MLP evaluation: {e}")
        return None

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
    print("🚀 Starting Comprehensive Model Evaluation")
    print("=" * 60)
    
    # Load the data
    print("📊 Loading GasProperties.csv dataset...")
    try:
        df = pd.read_csv('GasProperties.csv')
        print(f"✓ Loaded {len(df)} samples")
        
        # Optional subsampling for faster testing (uncomment if needed)
        # df = df.sample(n=10000, random_state=42)
        # print(f"✓ Subsampled to {len(df)} samples for faster testing")
        
    except FileNotFoundError:
        print("❌ Error: GasProperties.csv not found!")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Extract features and target
    X = df[['T', 'P', 'TC', 'SV']].values  # Features: T, P, TC, SV
    y = df['Idx'].values  # Target: Idx
    
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print(f"✓ Target range: {y.min():.2f} to {y.max():.2f}")
    
    # Preprocess the data
    print("\n🔧 Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess(X, y)
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    # Evaluate models for different polynomial orders
    max_order = 5
    print(f"\n🧮 Evaluating models for polynomial orders 1 to {max_order}")
    print("=" * 60)
    
    # Store results for comparison
    all_results = {}
    
    for order in range(1, max_order + 1):
        print(f"\n📈 POLYNOMIAL ORDER {order}")
        print("-" * 30)
        
        # Generate polynomial features
        X_train_poly = add_polynomial_features(X_train, order)
        X_test_poly = add_polynomial_features(X_test, order)
        print(f"✓ Generated polynomial features: {X_train_poly.shape[1]} features")
        
        # 1. Evaluate Least Squares
        print("\n🔵 Least Squares Model:")
        try:
            ls_results = evaluate_least_squares(X_train, X_test, y_train, y_test, max_order=order)
            if order in ls_results:
                ls_result = ls_results[order]
                print(f"  Train RMSE: {ls_result['train_rmse']:.6f}")
                print(f"  Test RMSE:  {ls_result['test_rmse']:.6f}")
                print(f"  Train R²:   {ls_result['train_r2']:.6f}")
                print(f"  Test R²:    {ls_result['test_r2']:.6f}")
                print(f"  Time:       {ls_result['training_time']:.4f}s")
                all_results[f'LS_Order_{order}'] = ls_result
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # 2. Evaluate Gradient Descent
        print("\n🟡 Gradient Descent Model:")
        try:
            gd_results = evaluate_gradient_descent(X_train, X_test, y_train, y_test, max_order=order)
            if order in gd_results:
                gd_result = gd_results[order]
                print(f"  Train RMSE: {gd_result['train_rmse']:.6f}")
                print(f"  Test RMSE:  {gd_result['test_rmse']:.6f}")
                print(f"  Train R²:   {gd_result['train_r2']:.6f}")
                print(f"  Test R²:    {gd_result['test_r2']:.6f}")
                print(f"  Time:       {gd_result['training_time']:.4f}s")
                all_results[f'GD_Order_{order}'] = gd_result
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # 3. Evaluate MLP
        print("\n🟣 MLP (Neural Network) Model:")
        try:
            mlp_result = evaluate_mlp(X_train_poly, X_test_poly, y_train, y_test, order)
            if mlp_result:
                print(f"  Train RMSE: {mlp_result['train_rmse']:.6f}")
                print(f"  Test RMSE:  {mlp_result['test_rmse']:.6f}")
                print(f"  Train R²:   {mlp_result['train_r2']:.6f}")
                print(f"  Test R²:    {mlp_result['test_r2']:.6f}")
                print(f"  Time:       {mlp_result['training_time']:.4f}s")
                all_results[f'MLP_Order_{order}'] = mlp_result
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    
    if all_results:
        print("\nBest performing models by Test R²:")
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
        for i, (model_name, result) in enumerate(sorted_results[:5]):  # Top 5
            print(f"{i+1}. {model_name}: R² = {result['test_r2']:.6f}, RMSE = {result['test_rmse']:.6f}")
    
    print("\n🎉 Evaluation completed successfully!")
    print("\n💡 Key Insights:")
    print("- Compare how different models perform on the same data")
    print("- See how polynomial features affect performance")
    print("- Notice the trade-off between model complexity and performance")
    print("- Observe training times for different approaches")

if __name__ == "__main__":
    run_tests()
