import torch as t
import torch.nn as nn
import numpy as np
from typeguard import typechecked

# Add your own imports here

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model for regression tasks.

    This class defines a feedforward neural network with a configurable number of
    hidden layers and neurons, using ReLU activation functions between layers.

    Attributes:
        layers (nn.Sequential): The sequential container of layers forming the MLP.
    """
    @typechecked
    def __init__(self, input_size: int, hidden_layers: list[int], output_size: int) -> None:
        """
        Initializes the MLP model.

        Args:
            input_size (int): The number of input features.
            hidden_layers (list[int]): A list where each element represents the number of neurons
                                        in a corresponding hidden layer.
            output_size (int): The number of output features.
        """
        super(MLP, self).__init__()
        
        # Build the network architecture
        layers = []
        
        # Input layer to first hidden layer
        if hidden_layers:
            layers.append(nn.Linear(input_size, hidden_layers[0]))
            layers.append(nn.ReLU())
            
            # Hidden layers
            for i in range(len(hidden_layers) - 1):
                layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
                layers.append(nn.ReLU())
            
            # Output layer
            layers.append(nn.Linear(hidden_layers[-1], output_size))
        else:
            # No hidden layers, direct input to output
            layers.append(nn.Linear(input_size, output_size))
        
        self.layers = nn.Sequential(*layers)

    @typechecked
    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Performs the forward pass of the MLP.

        Args:
            x (t.Tensor): The input tensor.

        Returns:
            t.Tensor: The output tensor from the MLP.
        """
        return self.layers(x)
    
@typechecked
def setup_model(input_size: int, hidden_layers: list[int], output_size: int, device: t.device) -> tuple[MLP, nn.Module, t.optim.Optimizer]:
    """
    Sets up the MLP model, loss function, and optimizer.

    Args:
        input_size (int): The number of input features for the MLP.
        hidden_layers (list[int]): A list defining the architecture of hidden layers.
        output_size (int): The number of output features for the MLP.
        device (t.device): The device (CPU or GPU) to deploy the model to.

    Returns:
        tuple[MLP, nn.Module, t.optim.Optimizer]: A tuple containing:
            - model (MLP): The initialized MLP model.
            - criterion (nn.Module): The loss function.
            - optimizer (t.optim.Optimizer): The optimizer for updating model parameters.
    """
    # Create the model
    model = MLP(input_size, hidden_layers, output_size)
    model = model.to(device)
    
    # Define loss function (MSE for regression)
    criterion = nn.MSELoss()
    
    # Define optimizer (Adam with reasonable default learning rate)
    optimizer = t.optim.Adam(model.parameters(), lr=0.001)
    
    return model, criterion, optimizer

@typechecked
def train_model(model: MLP, train_loader: t.utils.data.DataLoader, criterion: nn.Module, optimizer: t.optim.Optimizer, num_epochs: int) -> MLP:
    """
    Trains the MLP model using the provided data loader, criterion, and optimizer.

    Args:
        model (MLP): The MLP model to be trained.
        train_loader (t.utils.data.DataLoader): The data loader providing training batches.
        criterion (nn.Module): The loss function.
        optimizer (t.optim.Optimizer): The optimizer for updating model parameters.
        num_epochs (int): The number of training epochs.

    Returns:
        MLP: The trained MLP model.
    """
    model.train()  # Set model to training mode
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0 or epoch == 0:
            avg_loss = total_loss / num_batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}')
    
    return model
