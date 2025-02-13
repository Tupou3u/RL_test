import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from .init_model_gru import GRUAgent

class PPO_GRU:
    def __init__(self, 
                 state_dim:     int,
                 action_dim:    int,
                 learning_rate: float = 0.001, 
                 batch_size:    int = 32,
                 train_epochs:  int = 10, 
                 gamma:         float = 0.99, 
                 gae_lambda:    float = 0.95, 
                 actor_clip:    float = 0.1, 
                 critic_clip:   float = 0.1, 
                 ent_coef:      float = 0.0, 
                 alpha:         float = 0.1, 
                 device:        str = None, 
                 log_dir:       str = None,
                 agent_id:      int = None):

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.actor_clip = actor_clip
        self.critic_clip = critic_clip
        self.ent_coef = ent_coef
        self.alpha = alpha
        self.log_dir = log_dir
        self.agent_id = agent_id

        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

        if log_dir:
            if agent_id:
                self.log_dir = log_dir + f'agent_{agent_id}/'

            self.writer = SummaryWriter(self.log_dir)

        self.model = GRUAgent(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)
        self.num_trains = 0
    
    def get_advantages_gae(self, rewards, values):
        advantages = []
        for ep_rewards, ep_values in zip(rewards, values):
            ep_len = len(ep_rewards)
            adv = torch.zeros(ep_len).to('cpu')
            last_gae_lambda = 0
            
            for t in reversed(range(ep_len)):
                if t == ep_len - 1:
                    next_value = 0
                else:
                    next_value = ep_values[t + 1]
                
                delta = ep_rewards[t] + self.gamma * next_value - ep_values[t]
                adv[t] = last_gae_lambda = delta + self.gamma * self.gae_lambda * last_gae_lambda
                
            advantages.append(adv)
        
        return advantages
    
    def get_returns(self, advantages, values):
        returns = []
        for ep_advantages, ep_values in zip(advantages, values):
            returns.append(ep_advantages + ep_values)
        
        return returns

    def train(self, states, actions, rewards, logprobs, values, action_masks, sequences_lengths):
        advantages = self.get_advantages_gae(rewards, values)
        y = self.get_returns(advantages, values)

        rollout_len = len(states)
        inds = np.arange(rollout_len)

        init_gru_states = torch.zeros(self.model.gru.num_layers, self.batch_size, self.model.gru.hidden_size).to(self.device)

        clipfracs = []
        for _ in range(self.train_epochs):
            np.random.shuffle(inds)
            for start in range(0, rollout_len, self.batch_size):
                end = start + self.batch_size
                b_inds = inds[start:end]

                b_states = [states[i] for i in b_inds]
                b_sequences_lengths = [sequences_lengths[i] for i in b_inds]
                b_actions = [actions[i] for i in b_inds]
                b_action_masks = [action_masks[i] for i in b_inds]

                _, b_newlogprobs, b_entropys, b_newvalues, _ = self.model.get_actions_and_values(
                        b_states,
                        init_gru_states,
                        b_sequences_lengths,
                        b_actions,
                        b_action_masks,
                        self.device
                )

                b_logprobs = torch.cat([logprobs[i] for i in b_inds], dim=0).to(self.device)
                
                log_r = b_newlogprobs - b_logprobs
                r = log_r.exp()

                with torch.no_grad():
                    old_approx_kl = (-log_r).mean()
                    approx_kl = ((r - 1) - log_r).mean()
                    clipfracs += [((r - 1.0).abs() > self.actor_clip).float().mean().item()]

                b_advantages = torch.cat([advantages[i] for i in b_inds], dim=0).to(self.device)
                norm_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                p_loss1 = -norm_advantages * r
                p_loss2 = -norm_advantages * torch.clamp(r, 1 - self.actor_clip, 1 + self.actor_clip)
                p_loss = torch.max(p_loss1, p_loss2).mean()

                b_values = torch.cat([values[i] for i in b_inds], dim=0).to(self.device)
                b_y = torch.cat([y[i] for i in b_inds], dim=0).to(self.device)

                v_loss_unclipped = (b_newvalues - b_y) ** 2
                v_clipped = b_values + torch.clamp(
                    b_newvalues - b_values,
                    -self.critic_clip,
                    self.critic_clip,
                )
                v_loss_clipped = (v_clipped - b_y) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = v_loss_max.mean()

                entropy = b_entropys.mean()
                entropy_loss = - self.ent_coef * entropy
                loss = p_loss + self.ent_coef * entropy_loss + v_loss * self.alpha

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        if self.log_dir:
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.num_trains)
            self.writer.add_scalar("losses/critic_loss", v_loss.item(), self.num_trains)
            self.writer.add_scalar("losses/actor_loss", p_loss.item(), self.num_trains)
            self.writer.add_scalar("losses/entropy", entropy.item(), self.num_trains)
            self.writer.add_scalar("losses/entropy_loss", entropy_loss.item(), self.num_trains)
            self.writer.add_scalar("losses/loss", loss.item(), self.num_trains)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.num_trains)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.num_trains)
            self.num_trains += 1    
