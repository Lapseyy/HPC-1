#!/usr/bin/env python3
"""
Simple test file to demonstrate how to test the MLP functions.
This shows you the step-by-step process of testing your code.
"""

import torch as t
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from mlp import MLP, setup_model, train_model

def test_mlp_basic():
    """Test basic MLP creation and forward pass"""
    print("=== Testing Basic MLP Creation ===")
    
    # Create a simple MLP: 4 inputs -> [10, 5] hidden layers -> 1 output
    input_size = 4
    hidden_layers = [10, 5]
    output_size = 1
    
    # Create the model
    model = MLP(input_size, hidden_layers, output_size)
    print(f"✓ MLP created successfully")
    print(f"  Input size: {input_size}")
    print(f"  Hidden layers: {hidden_layers}")
    print(f"  Output size: {output_size}")
    
    # Test forward pass with random data
    batch_size = 3
    x = t.randn(batch_size, input_size)  # Random input data
    print(f"✓ Input tensor shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    print(f"✓ Output tensor shape: {output.shape}")
    print(f"✓ Forward pass successful!")
    
    return model

def test_setup_model():
    """Test the setup_model function"""
    print("\n=== Testing setup_model Function ===")
    
    input_size = 4
    hidden_layers = [10, 5]
    output_size = 1
    device = t.device('cpu')  # Use CPU for testing
    
    # Test setup_model
    model, criterion, optimizer = setup_model(input_size, hidden_layers, output_size, device)
    
    print(f"✓ Model created: {type(model).__name__}")
    print(f"✓ Loss function: {type(criterion).__name__}")
    print(f"✓ Optimizer: {type(optimizer).__name__}")
    print(f"✓ Model device: {next(model.parameters()).device}")
    
    return model, criterion, optimizer

def test_training():
    """Test the training function with synthetic data"""
    print("\n=== Testing Training Function ===")
    
    # Create synthetic data (like the gas properties data)
    np.random.seed(42)  # For reproducible results
    n_samples = 100
    n_features = 4
    
    # Generate synthetic features (like T, P, TC, SV from your CSV)
    X = np.random.randn(n_samples, n_features)
    
    # Generate synthetic target (like Idx from your CSV)
    # Make it depend on the features in a non-linear way
    y = (X[:, 0] * 2 + X[:, 1] * 3 + X[:, 2] * 0.5 + X[:, 3] * 1.5 + 
         X[:, 0] * X[:, 1] * 0.1 + np.random.randn(n_samples) * 0.1).reshape(-1, 1)
    
    print(f"✓ Generated synthetic data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✓ Target shape: {y.shape}")
    
    # Convert to PyTorch tensors
    X_tensor = t.FloatTensor(X)
    y_tensor = t.FloatTensor(y)
    
    # Create data loader
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"✓ Created data loader with batch size 16")
    
    # Setup model
    model, criterion, optimizer = setup_model(n_features, [10, 5], 1, t.device('cpu'))
    
    # Train the model
    print("✓ Starting training...")
    trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=50)
    
    # Test predictions
    model.eval()  # Set to evaluation mode
    with t.no_grad():
        test_input = t.randn(5, n_features)
        predictions = trained_model(test_input)
        print(f"✓ Test predictions shape: {predictions.shape}")
        print(f"✓ Sample predictions: {predictions.flatten()[:3].tolist()}")
    
    return trained_model

def test_with_real_data_sample():
    """Test with a small sample of real data"""
    print("\n=== Testing with Real Data Sample ===")
    
    try:
        import pandas as pd
        
        # Load a small sample of the real data
        print("Loading real data sample...")
        df = pd.read_csv('GasProperties.csv', nrows=1000)  # Only first 1000 rows
        
        # Extract features (T, P, TC, SV) and target (Idx)
        X = df[['T', 'P', 'TC', 'SV']].values
        y = df['Idx'].values.reshape(-1, 1)
        
        print(f"✓ Loaded real data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"✓ Target range: {y.min():.2f} to {y.max():.2f}")
        
        # Convert to PyTorch tensors
        X_tensor = t.FloatTensor(X)
        y_tensor = t.FloatTensor(y)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Setup and train model
        model, criterion, optimizer = setup_model(4, [20, 10], 1, t.device('cpu'))
        
        print("✓ Training with real data...")
        trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=100)
        
        # Test predictions
        model.eval()
        with t.no_grad():
            sample_input = X_tensor[:5]  # First 5 samples
            predictions = trained_model(sample_input)
            actual = y_tensor[:5]
            
            print(f"✓ Sample predictions vs actual:")
            for i in range(5):
                print(f"  Sample {i+1}: Predicted={predictions[i].item():.2f}, Actual={actual[i].item():.2f}")
        
    except FileNotFoundError:
        print("⚠ GasProperties.csv not found, skipping real data test")
    except Exception as e:
        print(f"⚠ Error with real data test: {e}")

def main():
    """Run all tests"""
    print("🧪 Testing MLP Implementation")
    print("=" * 50)
    
    try:
        # Test 1: Basic MLP creation
        test_mlp_basic()
        
        # Test 2: Setup model function
        test_setup_model()
        
        # Test 3: Training with synthetic data
        test_training()
        
        # Test 4: Training with real data sample
        test_with_real_data_sample()
        
        print("\n🎉 All tests completed successfully!")
        print("\n💡 How to use this knowledge:")
        print("1. Start with simple tests (like test_mlp_basic)")
        print("2. Test each function individually")
        print("3. Use synthetic data first, then real data")
        print("4. Print intermediate results to understand what's happening")
        print("5. Look at the shapes of tensors and data")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("This is normal! Debugging is part of learning.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
