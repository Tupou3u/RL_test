import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, action_dim), std=1.0)
        )

    def forward(self, x):
        return self.net(x)
    
    def get_action(self, x):
        logits = self.net(x)
        return torch.argmax(logits)