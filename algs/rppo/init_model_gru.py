import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

LAYER_SIZE = 128
DETACH = True

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class GRUAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, LAYER_SIZE)),
            nn.ReLU(),
            layer_init(nn.Linear(LAYER_SIZE, LAYER_SIZE)),
            nn.ReLU(),
        )
        self.gru = nn.GRU(LAYER_SIZE, LAYER_SIZE)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
                
        self.actor = layer_init(nn.Linear(LAYER_SIZE, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(LAYER_SIZE, 1), std=1)

    def get_states(self, x, gru_state, done):
        hidden = self.network(x)
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
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)

        return new_hidden, gru_state

    def get_value(self, x, gru_state, done):
        hidden, _ = self.get_states(x, gru_state, done)
        if DETACH:
            return self.critic(hidden.detach())
        return self.critic(hidden)

    def get_action_and_value(self, x, gru_state, done, action=None):
        hidden, gru_state = self.get_states(x, gru_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        if DETACH:
            value = self.critic(hidden.detach())
        else:
            value = self.critic(hidden)
            
        return action, value, gru_state, probs.log_prob(action), probs.entropy()
    
if __name__ == '__main__':
    model = GRUAgent(5, 5)
    print(model.state_dict()['gru.weight_hh_l0'])