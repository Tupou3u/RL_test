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
        
        self.states = []
        self.action_masks = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.next_action_masks = []
        self.dones = []
        
    def add(self, game_states, game_action_masks, game_actions, game_rewards, game_next_states, game_next_action_masks, game_dones):
        self.states.append(game_states)
        self.action_masks.append(game_action_masks)
        self.actions.append(game_actions)
        self.rewards.append(game_rewards)
        self.next_states.append(game_next_states)
        self.next_action_masks.append(game_next_action_masks)
        self.dones.append(game_dones)
        
        self.states = self.states[-self.buffer_size:]
        self.action_masks = self.action_masks[-self.buffer_size:]
        self.actions = self.actions[-self.buffer_size:]
        self.rewards = self.rewards[-self.buffer_size:]
        self.next_states = self.next_states[-self.buffer_size:]
        self.next_action_masks = self.next_action_masks[-self.buffer_size:]
        self.dones = self.dones[-self.buffer_size:]

    def sample(self, batch_size):
        b_inds = np.random.choice(len(self.states), batch_size, replace=False)
        b_states = [torch.tensor(self.states[i], dtype=torch.float32) for i in b_inds]
        b_action_masks = [torch.tensor(self.action_masks[i], dtype=torch.float32) for i in b_inds]
        b_actions = [torch.tensor(self.actions[i], dtype=torch.float32) for i in b_inds]
        b_rewards = [torch.tensor(self.rewards[i], dtype=torch.float32) for i in b_inds]
        b_next_states = [torch.tensor(self.next_states[i], dtype=torch.float32) for i in b_inds]
        b_next_action_masks = [torch.tensor(self.next_action_masks[i], dtype=torch.float32) for i in b_inds]
        b_dones = [torch.tensor(self.dones[i], dtype=torch.float32) for i in b_inds]

        b_states = torch.cat(b_states).to(self.device)
        b_action_masks = torch.cat(b_action_masks).to(self.device)
        b_actions = torch.cat(b_actions).to(self.device)
        b_rewards = torch.cat(b_rewards).to(self.device)
        b_next_states = torch.cat(b_next_states).to(self.device)
        b_next_action_masks = torch.cat(b_next_action_masks).to(self.device)
        b_dones = torch.cat(b_dones).to(self.device)

        return b_states, b_action_masks, b_actions.unsqueeze(1), b_rewards.unsqueeze(1), b_next_states, b_next_action_masks, b_dones.unsqueeze(1)

    def __len__(self):
        return len(self.states)