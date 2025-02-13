import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchrl.data import PrioritizedReplayBuffer


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

    def get_action(self, x):
        logits = self.actor(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=-1)
        return action, log_prob, action_probs

class PER_SACAgentMLP:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            policy_lr: float = 1e-4,
            q_lr: float = 1e-4,
            batch_size: int = 1000,
            train_steps: int = 50,
            gamma: float = 0.99,
            tau: float = 0.995,
            alpha: float = 0.01,
            autotune: bool = False,
            target_entropy_scale: float = 0.01,
            device: str = 'cpu',
            log_dir: str = None
        ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_lr = policy_lr
        self.q_lr = q_lr
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
            self.target_entropy = -target_entropy_scale * torch.log(1 / torch.tensor(action_dim))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=q_lr, eps=1e-4)
        else:
            self.alpha = alpha

        if self.log_dir:    
            self.writer = SummaryWriter(self.log_dir) 
            self.num_trains = 0

    def train_step(self, replayBuffer: PrioritizedReplayBuffer):        
        b_data, b_info = replayBuffer.sample(self.batch_size, return_info=True)
        b_states = b_data['state'].float().to(self.device) 
        b_actions = b_data['action'].float().to(self.device)
        b_rewards = b_data['reward'].float().to(self.device)
        b_next_states = b_data['next_state'].float().to(self.device)
        b_dones = b_data['done'].float().to(self.device)
        b_weights = b_info['_weight'].float().to(self.device)

        # CRITIC training
        _, next_state_log_pi, next_state_action_probs = self.actor.get_action(
            b_next_states
        )
        with torch.no_grad():
            qf1_next_target = self.qf1_target(b_next_states)
            qf2_next_target = self.qf2_target(b_next_states)
            min_qf_next_target = next_state_action_probs * (
                torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            )
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(1)
            next_q_value = b_rewards.reshape(-1, 1) + (1 - b_dones.reshape(-1, 1)) * self.gamma * (min_qf_next_target)

        qf1_values = self.qf1(b_states)
        qf2_values = self.qf2(b_states)
        qf1_a_values = qf1_values.gather(1, b_actions.reshape(-1, 1).long())
        qf2_a_values = qf2_values.gather(1, b_actions.reshape(-1, 1).long())
        qf1_loss = (b_weights * F.mse_loss(qf1_a_values, next_q_value, reduction='none')).mean()
        qf2_loss = (b_weights * F.mse_loss(qf2_a_values, next_q_value, reduction='none')).mean()
        qf_loss = qf1_loss + qf2_loss

        td_error_1 = qf1_a_values - next_q_value
        td_error_2 = qf2_a_values - next_q_value
        td_error = torch.abs(td_error_1 + td_error_2) / 2.0
        replayBuffer.update_priority(b_info['index'], td_error)

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # ACTOR training
        _, log_pi, action_probs = self.actor.get_action(
            b_states
        )
        with torch.no_grad():
            qf1_values = self.qf1(b_states)
            qf2_values = self.qf2(b_states)
            min_qf_values = torch.min(qf1_values, qf2_values)

        actor_loss = (action_probs * ((self.alpha * log_pi) - min_qf_values)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.autotune:
            alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach())).mean()
            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            
        # update target networks
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.log_dir:
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

    def train(self, replayBuffer: PrioritizedReplayBuffer):
        if len(replayBuffer) < self.batch_size:
            raise Exception('Not enough samples in replay buffer')
        
        for _ in range(self.train_steps):
            self.train_step(replayBuffer)


        


        