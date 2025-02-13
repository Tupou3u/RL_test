import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

LAYER_SIZE = 128

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MLPAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, LAYER_SIZE)),
            nn.ReLU(),
            layer_init(nn.Linear(LAYER_SIZE, LAYER_SIZE)),
            nn.ReLU(),
        )                
        self.actor = layer_init(nn.Linear(LAYER_SIZE, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(LAYER_SIZE, 1), std=1)

    def get_action(self, x):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        return probs.sample()
    
    def get_value(self, x):
        hidden = self.network(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
     
        return action, self.critic(hidden).squeeze(), probs.log_prob(action), probs.entropy()


class PPO_MLP:
    def __init__(self, 
            state_dim: int = None,
            action_dim: int = None,
            learning_rate: float = 0.001, 
            batch_size: int = 32,
            train_epochs: int = 10, 
            gamma: float = 0.99, 
            gae_lambda: float = 0.95, 
            actor_clip: float = 0.1, 
            critic_clip: float = 0.1, 
            ent_coef: float = 0.001, 
            alpha: float = 0.1, 
            device: str = 'cpu', 
            log_dir: str = None
        ):

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.actor_clip = actor_clip
        self.critic_clip = critic_clip
        self.ent_coef = ent_coef
        self.alpha = alpha
        self.device = device
        self.log_dir = log_dir

        if log_dir:
            self.writer = SummaryWriter(self.log_dir)

        self.model = MLPAgent(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)
        self.num_trains = 0
    
    def __get_adv_gae(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(rewards.shape[0] - 1)):
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        
        return advantages

    def train(self, states, actions, rewards, dones, logprobs, values):
        adv_gae = self.__get_adv_gae(rewards, values, dones)
        y = adv_gae + values[:-1]

        y_pred, y_true = values[:-1].cpu().numpy(), y.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        states = states.reshape(-1, states.shape[-1])
        actions = actions.reshape(-1)
        logprobs = logprobs.reshape(-1)
        values = values.reshape(-1)
        adv_gae = adv_gae.reshape(-1)
        y = y.reshape(-1)

        rollout_length = len(states)
        inds = np.arange(rollout_length)
        clipfracs = []

        for _ in range(self.train_epochs):
            np.random.shuffle(inds)
            for start in range(0, rollout_length, self.batch_size):
                end = start + self.batch_size
                mb_inds = inds[start:end]
                
                _, newvalue, newlogprob, entropy = self.model.get_action_and_value(
                        states[mb_inds],
                        actions[mb_inds],
                )

                log_r = newlogprob - logprobs[mb_inds]
                r = log_r.exp()

                with torch.no_grad():
                    old_approx_kl = (-log_r).mean()
                    approx_kl = ((r - 1) - log_r).mean()
                    clipfracs += [((r - 1.0).abs() > self.actor_clip).float().mean().item()]

                norm_adv = (adv_gae[mb_inds] - adv_gae[mb_inds].mean()) / (adv_gae[mb_inds].std() + 1e-8)

                p_loss1 = -norm_adv * r
                p_loss2 = -norm_adv * torch.clamp(r, 1 - self.actor_clip, 1 + self.actor_clip)
                p_loss = torch.max(p_loss1, p_loss2).mean()

                v_loss_unclipped = (newvalue - y[mb_inds]) ** 2
                v_clipped = values[mb_inds] + torch.clamp(
                    newvalue - values[mb_inds],
                    -self.critic_clip,
                    self.critic_clip,
                )
                v_loss_clipped = (v_clipped - y[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = p_loss - self.ent_coef * entropy_loss + v_loss * self.alpha

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        if self.log_dir:
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.num_trains)
            self.writer.add_scalar("losses/explained_variance", explained_var, self.num_trains)
            self.writer.add_scalar("losses/critic_loss", v_loss.item(), self.num_trains)
            self.writer.add_scalar("losses/actor_loss", p_loss.item(), self.num_trains)
            self.writer.add_scalar("losses/entropy_loss", entropy_loss.item(), self.num_trains)
            self.writer.add_scalar("losses/loss", loss.item(), self.num_trains)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.num_trains)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.num_trains)
        self.num_trains += 1
