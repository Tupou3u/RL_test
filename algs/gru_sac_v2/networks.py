import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    

class GRUSoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
        )
        self.gru = nn.GRU(128, 128)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1e-6)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.critic = layer_init(nn.Linear(128, action_dim), std=1.0)
    
    def get_value(self, x, gru_state):
        hidden = self.network(x)
        batch_size = gru_state.shape[1]
        hidden = hidden.reshape((-1, batch_size, self.gru.input_size))
        new_hidden, gru_state = self.gru(hidden, gru_state)
        return self.critic(new_hidden), gru_state
    
    def get_values(self, sequences, gru_states, s_lenghts, device):
        padded_sequences = pad_sequence(sequences, batch_first=True).to(device)
        hidden = self.network(padded_sequences)
        packed_sequences = pack_padded_sequence(hidden, s_lenghts, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_sequences, gru_states)
        new_hidden, _ = pad_packed_sequence(packed_output, batch_first=True)
        all_values = self.critic(new_hidden)
        values = [all_values[i, :l] for i, l in enumerate(s_lenghts)]
        return values, gru_states


class GRUActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
        )
        self.gru = nn.GRU(128, 128)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1e-6)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(128, action_dim), std=0.01)
    
    def get_action(self, x, gru_state):
        hidden = self.network(x)
        batch_size = gru_state.shape[1]
        hidden = hidden.reshape((-1, batch_size, self.gru.input_size))
        new_hidden, gru_state = self.gru(hidden, gru_state)
        logits = self.actor(new_hidden)
        probs = Categorical(logits=logits)
        action = probs.sample()
        action_probs = probs.probs
        log_prob = F.log_softmax(logits, dim=-1)
        return action, gru_state, log_prob, action_probs, probs.entropy()
    
    def get_actions(self, sequences, gru_states, s_lenghts, device):
        padded_sequences = pad_sequence(sequences, batch_first=True).to(device)
        hidden = self.network(padded_sequences)
        packed_sequences = pack_padded_sequence(hidden, s_lenghts, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_sequences, gru_states)
        new_hidden, _ = pad_packed_sequence(packed_output, batch_first=True)
        all_logits = self.actor(new_hidden)
        probs = Categorical(logits=all_logits)
        all_actions = probs.sample()
        all_action_probs = probs.probs
        all_log_probs = F.log_softmax(all_logits, dim=-1)

        actions, log_probs, action_probs = [], [], []
        for i, l in enumerate(s_lenghts):          
            actions.append(all_actions[i, :l])
            log_probs.append(all_log_probs[i, :l])
            action_probs.append(all_action_probs[i, :l])

        return actions, gru_states, log_probs, action_probs, probs.entropy()
