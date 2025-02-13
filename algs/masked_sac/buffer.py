import numpy as np
import torch

class ReplayBuffer:
    def __init__(
            self, 
            buffer_size,
            state_dim, 
            action_dim,
            device
        ):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        self.states = np.zeros((self.buffer_size, state_dim), dtype=np.float32)
        self.action_masks = np.zeros((self.buffer_size, action_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros_like(self.states, dtype=np.float32)
        self.next_action_masks = np.zeros((self.buffer_size, action_dim), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0

    def add(self, state, action_mask, action, reward, next_state, next_action_mask, done):
        self.states[self.ptr] = state
        self.action_masks[self.ptr] = action_mask
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.next_action_masks[self.ptr] = next_action_mask
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        states = torch.Tensor(self.states[indices]).to(self.device)
        action_masks = torch.Tensor(self.action_masks[indices]).to(self.device)
        actions = torch.Tensor(self.actions[indices]).to(self.device)
        rewards = torch.Tensor(self.rewards[indices]).to(self.device)
        next_states = torch.Tensor(self.next_states[indices]).to(self.device)
        next_action_masks = torch.Tensor(self.next_action_masks[indices]).to(self.device)
        dones = torch.Tensor(self.dones[indices]).to(self.device)

        return states, action_masks, actions, rewards, next_states, next_action_masks, dones

    def __len__(self):
        return self.size