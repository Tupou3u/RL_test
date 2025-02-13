import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

env_id: str = "CartPole-v1"
total_epochs: int = 5000000
buffer_size: int = int(1e6)
gamma: float = 0.99
tau: float = 0.99
batch_size: int = 100
learning_starts: int = int(1e4)
policy_lr: float = 0.001
q_lr: float = 0.001
update_frequency: int = 10
# target_network_frequency: int = 100
alpha: float = 0.1
autotune: bool = False
target_entropy_scale: float = 0.89
device = 'cpu'
log_interval = 100
train_steps = 50

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = env.observation_space.shape[0]
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, env.action_space.n))
        )

    def forward(self, x):
        return self.critic(x)

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = env.observation_space.shape[0]
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, env.action_space.n)),
        )

    def forward(self, x):
        return self.actor(x)

    def get_action(self, x):
        logits = self.actor(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=-1)
        return action, log_prob, action_probs
    
env = gym.make(env_id)

save_path = 'clean_rl/history/' + time.strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(save_path)

actor = Actor(env).to(device)
qf1 = SoftQNetwork(env).to(device)
qf2 = SoftQNetwork(env).to(device)
qf1_target = SoftQNetwork(env).to(device)
qf2_target = SoftQNetwork(env).to(device)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=q_lr, eps=1e-4)
actor_optimizer = optim.Adam(list(actor.parameters()), lr=policy_lr, eps=1e-4)

if autotune:
    target_entropy = -target_entropy_scale * torch.log(1 / torch.tensor(env.action_space.n))
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=q_lr, eps=1e-4)
else:
    alpha = alpha

rb = ReplayBuffer(
    buffer_size,
    env.observation_space,
    env.action_space,
    device,
    handle_timeout_termination=False,
)
start_time = time.time()
mean_reward, mean_steps = 0, 0

for epoch in range(total_epochs):
    obs, _ = env.reset()
    termination, truncated = False, False
    ep_reward, ep_steps = 0, 0
    while not (termination or truncated):
        action, _, _ = actor.get_action(torch.Tensor(obs).to(device))
        next_obs, reward, termination, truncated, _ = env.step(action.detach().cpu().numpy())
        rb.add(obs, next_obs, action, reward, termination or truncated, None)
        obs = next_obs
        ep_reward += reward
        ep_steps += 1

        if termination or truncated:
            writer.add_scalar("charts/episodic_return", ep_reward, epoch)
            writer.add_scalar("charts/episodic_length", ep_steps, epoch)

    mean_reward += ep_reward
    mean_steps += ep_steps

    if epoch % log_interval == 0:
        print(f'epoch: {epoch} mean_reward: {round(mean_reward / log_interval, 2)} mean_steps: {round(mean_steps / log_interval, 2)}')
        mean_reward, mean_steps = 0, 0

    # ALGO LOGIC: training.
    if rb.size() > learning_starts and epoch % update_frequency == 0:
        for _ in range(train_steps):
            data = rb.sample(batch_size)
            # CRITIC training
            with torch.no_grad():
                _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations)
                qf2_next_target = qf2_target(data.next_observations)
                # we can use the action probabilities instead of MC sampling to estimate the expectation
                min_qf_next_target = next_state_action_probs * (
                    torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                )
                # adapt Q-target for discrete Q-function
                min_qf_next_target = min_qf_next_target.sum(dim=1)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * gamma * (min_qf_next_target)

            # use Q-values only for the taken actions
            qf1_values = qf1(data.observations)
            qf2_values = qf2(data.observations)
            qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
            qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # ACTOR training
            _, log_pi, action_probs = actor.get_action(data.observations)
            with torch.no_grad():
                qf1_values = qf1(data.observations)
                qf2_values = qf2(data.observations)
                min_qf_values = torch.min(qf1_values, qf2_values)
            # no need for reparameterization, the expectation can be calculated for discrete actions
            actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            if autotune:
                # re-use action probabilities for temperature loss
                alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                a_optimizer.zero_grad()
                alpha_loss.backward()
                a_optimizer.step()
                alpha = log_alpha.exp().item()

        for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), epoch)
        writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), epoch)
        writer.add_scalar("losses/qf1_loss", qf1_loss.item(), epoch)
        writer.add_scalar("losses/qf2_loss", qf2_loss.item(), epoch)
        writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, epoch)
        writer.add_scalar("losses/actor_loss", actor_loss.item(), epoch)
        writer.add_scalar("losses/alpha", alpha, epoch)
        writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), epoch)
        if autotune:
            writer.add_scalar("losses/alpha_loss", alpha_loss.item(), epoch)

env.close()