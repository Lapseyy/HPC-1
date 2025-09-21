#!/usr/bin/env python3
"""
A very simple test to show you how to figure out if your code works.
This is what you would do step by step to test your functions.
"""

# Step 1: Import what you need
import torch as t
import numpy as np
from mlp import MLP, setup_model, train_model

print("🔍 Let's test our MLP step by step...")
print()

# Step 2: Test the simplest thing first - can we create an MLP?
print("Step 1: Can we create an MLP?")
try:
    model = MLP(input_size=3, hidden_layers=[5], output_size=1)
    print("✅ Yes! MLP created successfully")
    print(f"   Model has {sum(p.numel() for p in model.parameters())} parameters")
except Exception as e:
    print(f"❌ No! Error: {e}")

print()

# Step 3: Can we do a forward pass?
print("Step 2: Can we do a forward pass?")
try:
    # Create some fake data (3 samples, 3 features each)
    fake_data = t.randn(3, 3)
    print(f"   Input shape: {fake_data.shape}")
    
    output = model(fake_data)
    print(f"   Output shape: {output.shape}")
    print("✅ Yes! Forward pass works")
except Exception as e:
    print(f"❌ No! Error: {e}")

print()

# Step 4: Can we set up the model with loss and optimizer?
print("Step 3: Can we set up model with loss and optimizer?")
try:
    model, loss_fn, optimizer = setup_model(input_size=3, hidden_layers=[5], output_size=1, device=t.device('cpu'))
    print("✅ Yes! Setup works")
    print(f"   Loss function: {type(loss_fn).__name__}")
    print(f"   Optimizer: {type(optimizer).__name__}")
except Exception as e:
    print(f"❌ No! Error: {e}")

print()

# Step 5: Can we train with very simple data?
print("Step 4: Can we train with simple data?")
try:
    # Create very simple training data
    X = t.randn(10, 3)  # 10 samples, 3 features
    y = t.randn(10, 1)  # 10 targets
    
    # Create a simple data loader
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    print("   Created simple training data")
    
    # Train for just 2 epochs
    trained_model = train_model(model, train_loader, loss_fn, optimizer, num_epochs=2)
    print("✅ Yes! Training works")
    
    # Test prediction
    test_input = t.randn(2, 3)
    prediction = trained_model(test_input)
    print(f"   Test prediction shape: {prediction.shape}")
    
except Exception as e:
    print(f"❌ No! Error: {e}")

print()
print("🎯 Key Learning Points:")
print("1. Start with the simplest test possible")
print("2. Test one function at a time")
print("3. Use fake/simple data first")
print("4. Print shapes and values to understand what's happening")
print("5. If something fails, look at the error message carefully")
print("6. Each test builds on the previous one")
