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
        # Define the layers of the MLP
        layers: list[nn.Module] = []
        in_features = input_size
        for hidden_units in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.ReLU())
            in_features = hidden_units
        layers.append(nn.Linear(in_features, output_size))
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
        # Pass the input through the layers
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
    # Initialize the MLP model and move it to the specified device
    model = MLP(input_size, hidden_layers, output_size).to(device)
    
    criterion = nn.MSELoss()
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
    # Set the model to training mode
    model.train()
    
    # Get the device the model is on
    device = next(model.parameters()).device
    
    # Determine the output feature size from the model
    last_linear = model.layers[-1]                  # final layer is Linear
    out_features = last_linear.out_features

    for _ in range(num_epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            # ensure y has shape (N, out_features)
            if y_batch.ndim == 1:
                y_batch = y_batch.view(-1, 1)
            if y_batch.shape[1] != out_features:
                # reshape if possible, else error
                y_batch = y_batch.view(-1, out_features)

            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
    
    return model