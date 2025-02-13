import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class GRUAgent(nn.Module):
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
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)

    def get_action_and_value(self, x, gru_state, action_mask):
        hidden = self.network(x)
        batch_size = gru_state.shape[1]
        hidden = hidden.reshape((-1, batch_size, self.gru.input_size))
        new_hidden, gru_state = self.gru(hidden, gru_state)
        logits = self.actor(new_hidden).squeeze(0)
        action_mask = action_mask.reshape((logits.shape[0], -1))
        logits[action_mask == 0] = -1e9
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden).view(gru_state.shape[1]), gru_state

    def get_actions_and_values(self, sequence, gru_state, sequence_length, action, action_mask, device):
        padded_sequences = pad_sequence(sequence, batch_first=True).to(device)
        hidden = self.network(padded_sequences)
        packed_sequences = pack_padded_sequence(hidden, sequence_length, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_sequences, gru_state)
        new_hidden, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        all_logits = self.actor(new_hidden)
        all_values = self.critic(new_hidden)

        max_len = all_logits.size(1)
        mask = torch.arange(max_len)[None, :] < output_lengths[:, None]
        mask = mask.to(device)
        masked_logits = all_logits.masked_fill(~mask.unsqueeze(-1), -1e9)

        if action_mask is not None:
            padded_action_mask = pad_sequence(action_mask, batch_first=True, padding_value=0)
            masked_logits[padded_action_mask == 0] = -1e9

        probs = Categorical(logits=masked_logits)

        padded_actions = pad_sequence(action, batch_first=True, padding_value=0).to(device)
        all_log_probs = probs.log_prob(padded_actions)
        all_entropies = probs.entropy()

        log_probs, entropies, values = [], [], []

        for i, length in enumerate(sequence_length):          
            log_probs.append(all_log_probs[i, :length])
            entropies.append(all_entropies[i, :length])
            values.append(all_values[i, :length])

        log_probs = torch.cat(log_probs, dim=0)
        entropies = torch.cat(entropies, dim=0)
        values = torch.cat(values, dim=0)

        return None, log_probs, entropies, values, None