import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module): # Inheriting all functionalities required to work with neural network layers
    def __init__(self, model_dim, ff_dim):
        super(FeedForwardNetwork, self).__init__()
        
        # Defining the linear layers for the feedforward network
        
        # Defining the duo of fully connected layers
        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, model_dim)
        
        # GELU gives smoother gradients near zero than ReLU, reducing dead neurons
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        
        # Applying the second fully connected layer
        x = self.fc2(x)
        
        return x
    