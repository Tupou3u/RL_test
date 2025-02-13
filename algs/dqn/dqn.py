from algs.dqn.net import QNetwork
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchrl.data import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

class DQNAgent:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            lr: float = 1e-4,
            gamma: float = 0.99,
            tau: float = 0.995,
            batch_size: int = 256,
            train_steps: int = 10,
            device: str = 'cpu',
            log_dir: str = None
    ):
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.device = device
        self.log_dir = log_dir

        if self.log_dir:    
            self.writer = SummaryWriter(self.log_dir) 
            self.num_trains = 0

        self.model = QNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.model.state_dict())

    def train_step(self, replayBuffer: ReplayBuffer):
        b_data = replayBuffer.sample(self.batch_size)
        b_states = b_data['state'].float().to(self.device) 
        b_actions = b_data['action'].to(self.device)
        b_rewards = b_data['reward'].float().to(self.device)
        b_next_states = b_data['next_state'].float().to(self.device)
        b_dones = b_data['done'].float().to(self.device)

        with torch.no_grad():
            target_max, _ = self.target_network(b_next_states).max(dim=1)
            td_target = b_rewards.flatten() + self.gamma * target_max * (1 - b_dones.flatten())

        old_val = self.model(b_states).gather(1, b_actions.reshape(-1, 1)).squeeze()
        loss = F.mse_loss(td_target, old_val)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_network_param, q_network_param in zip(self.target_network.parameters(), self.model.parameters()):
            target_network_param.data.copy_(
                self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
            )

        if self.log_dir:
            self.writer.add_scalar("losses/td_loss", loss, self.num_trains)
            self.writer.add_scalar("losses/q_values", old_val.mean().item(), self.num_trains)
            self.num_trains += 1

    def train(self, replayBuffer: ReplayBuffer):
        if len(replayBuffer) < self.batch_size:
            raise Exception('Not enough samples in replay buffer')
        
        for _ in range(self.train_steps):
            self.train_step(replayBuffer)

