import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
class GRUSoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU()
        )
        self.gru = nn.GRU(128, 128)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1e-5)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.critic = layer_init(nn.Linear(128, action_dim), std=1.0)
    
    def get_value(self, x, gru_state, done):
        hidden = self.net(x)
        batch_size = gru_state.shape[1]
        hidden = hidden.reshape((-1, batch_size, self.gru.input_size))
        done = done.reshape((-1, batch_size))

        new_hidden = []
        for h, d in zip(hidden, done):
            h, gru_state = self.gru(
                h.unsqueeze(0),
                (1.0 - d).view(1, -1, 1) * gru_state
            )
            new_hidden += [h]
        new_hidden = torch.cat(new_hidden)
        return self.critic(new_hidden), gru_state

class GRUActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU()
        )
        self.gru = nn.GRU(128, 128)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1e-5)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(128, action_dim), std=0.01)

    def get_states(self, x, gru_state, done):
        hidden = self.net(x)
        batch_size = gru_state.shape[1]
        hidden = hidden.reshape((-1, batch_size, self.gru.input_size))
        done = done.reshape((-1, batch_size))
        
        new_hidden = []
        for h, d in zip(hidden, done):
            h, gru_state = self.gru(
                h.unsqueeze(0),
                (1.0 - d).view(1, -1, 1) * gru_state
            )
            new_hidden += [h]
        new_hidden = torch.cat(new_hidden)
        return new_hidden, gru_state

    def get_action(self, x, gru_state, done):
        hidden, gru_state = self.get_states(x, gru_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        action = probs.sample()
        action_probs = probs.probs
        log_prob = F.log_softmax(logits, dim=-1)
        return action, gru_state, log_prob, action_probs, probs.entropy()
