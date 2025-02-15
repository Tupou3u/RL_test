import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchrl.data import ReplayBuffer

LAYER_SIZE = 128
LOG_STD_MAX = 2
LOG_STD_MIN = -5

class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(LAYER_SIZE, 1)
        )
    
    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        return self.critic(x)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):  
        self.action_scale = 1.0
        self.action_bias = 0.0  
        super().__init__()
        self.actor_head = nn.Sequential(
            nn.Linear(state_dim, LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(LAYER_SIZE, LAYER_SIZE),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(LAYER_SIZE, action_dim)
        self.actor_logstd = nn.Linear(LAYER_SIZE, action_dim)
    
    def forward(self, x):
        hidden = self.actor_head(x)
        action_mean = self.actor_mean(hidden)
        action_logstd = torch.tanh(self.actor_logstd(hidden))
        action_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (action_logstd + 1)
        return action_mean, action_logstd

    def get_action(self, x):
        action_mean, action_logstd = self(x)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        x_t = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(action_mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class SAC_Continuous:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            policy_lr: float = 1e-4,
            q_lr: float = 1e-4,
            a_lr: float = 1e-3,
            batch_size: int = 1000,
            train_steps: int = 50,
            gamma: float = 0.99,
            tau: float = 0.005,
            alpha: float = 0.01,
            autotune: bool = False,
            device: str = 'cpu',
            log_dir: str = None
        ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.a_lr = a_lr
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.gamma = gamma
        self.tau = tau
        self.autotune = autotune
        self.device = device
        self.log_dir = log_dir

        self.actor = Actor(state_dim, action_dim).to(device)
        self.qf1 = SoftQNetwork(state_dim, action_dim).to(device)
        self.qf2 = SoftQNetwork(state_dim, action_dim).to(device)
        self.qf1_target = SoftQNetwork(state_dim, action_dim).to(device)
        self.qf2_target = SoftQNetwork(state_dim, action_dim).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr, eps=1e-4)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=policy_lr, eps=1e-4)

        if self.autotune:
            self.target_entropy = -torch.tensor(action_dim).to(device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=a_lr)
        else:
            self.alpha = alpha

        if self.log_dir:    
            self.writer = SummaryWriter(self.log_dir) 
        self.num_trains = 0

    def train_step(self, replayBuffer: ReplayBuffer):        
        b_data = replayBuffer.sample(self.batch_size)
        b_states = b_data['state'].to(self.device) 
        b_actions = b_data['action'].to(self.device)
        b_rewards = b_data['reward'].to(self.device)
        b_next_states = b_data['next_state'].to(self.device)
        b_next_dones = b_data['next_done'].to(self.device)
        
        # CRITIC training
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(b_next_states)
            qf1_next_target = self.qf1_target(b_next_states, next_state_actions)
            qf2_next_target = self.qf2_target(b_next_states, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = b_rewards + (1 - b_next_dones) * self.gamma * min_qf_next_target.squeeze()

        qf1_a_values = self.qf1(b_states, b_actions).squeeze()
        qf2_a_values = self.qf2(b_states, b_actions).squeeze()
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # ACTOR training
        pi, log_pi, _ = self.actor.get_action(b_states)
        qf1_pi = self.qf1(b_states, pi)
        qf2_pi = self.qf2(b_states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.autotune:
            with torch.no_grad():
                _, log_pi, _ = self.actor.get_action(b_states)
            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            
        # update target networks
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.log_dir and self.num_trains % self.train_steps == 0:
            self.writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), self.num_trains)
            self.writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), self.num_trains)
            self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), self.num_trains)
            self.writer.add_scalar("losses/qf2_loss", qf2_loss.item(), self.num_trains)
            self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, self.num_trains)
            self.writer.add_scalar("losses/actor_loss", actor_loss.item(), self.num_trains)
            self.writer.add_scalar("losses/alpha", self.alpha, self.num_trains)
            if self.autotune:
                self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), self.num_trains)
                
        self.num_trains += 1

    def train(self, replayBuffer: ReplayBuffer):
        if len(replayBuffer) < self.batch_size:
            raise Exception('Not enough samples in replay buffer')
        
        for _ in range(self.train_steps):
            self.train_step(replayBuffer)


        


        