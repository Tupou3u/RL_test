from torchrl.data import ReplayBuffer
import torch
import random

class HER:
    def __init__(
            self, 
            prob: float = 0.2,
            k: int = 4,
            term_reward: float = 0.0,
            replay_buffer: ReplayBuffer = ReplayBuffer
        ):
        self.prob = prob
        self.k = k
        self.term_reward = term_reward
        self.rb = replay_buffer

    def add(self, obs, action, next_obs, next_ach):
        if random.random() < self.prob:
            data = {
                'state': torch.cat((obs, next_ach)),
                'action': action,
                'reward': torch.tensor(self.term_reward, dtype=torch.float32),
                'next_state': torch.cat((next_obs, next_ach)),
                'next_done': torch.tensor(1, dtype=torch.float32)
            }
            for _ in range(self.k):
                self.rb.add(data)
