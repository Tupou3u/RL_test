import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_dim), std=1.0)
        )
    
    def forward(self, x):
        return self.critic(x)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_dim), std=0.01),
        )

    def forward(self, x):
        return self.actor(x)

    def get_action(self, x, action_mask=None):
        logits = self.actor(x)

        if action_mask is not None:
            logits[action_mask == 0] = -1e9

        probs = Categorical(logits=logits)
        action = probs.sample()
        action_probs = probs.probs
        log_prob = F.log_softmax(logits, dim=-1)
        return action, log_prob, action_probs