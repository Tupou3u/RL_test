import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from algs.gru_sac_v2.networks import *
import random

class SACAgentGRU:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            policy_lr: float = 1e-4,
            q_lr: float = 1e-4,
            a_lr: float = 1e-3,
            batch_size: int = 100,
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
        self.a_lr = a_lr
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.gamma = gamma
        self.tau = tau
        self.autotune = autotune
        self.device = device
        self.log_dir = log_dir

        self.actor = GRUActor(state_dim, action_dim).to(device)
        self.qf1 = GRUSoftQNetwork(state_dim, action_dim).to(device)
        self.qf2 = GRUSoftQNetwork(state_dim, action_dim).to(device)
        self.qf1_target = GRUSoftQNetwork(state_dim, action_dim).to(device)
        self.qf2_target = GRUSoftQNetwork(state_dim, action_dim).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr, eps=1e-4)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=policy_lr, eps=1e-4)

        if self.autotune:
            self.target_entropy = -target_entropy_scale * torch.log(1 / torch.tensor(action_dim))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=a_lr, eps=1e-4)
        else:
            self.alpha = alpha

        if self.log_dir:    
            self.writer = SummaryWriter(self.log_dir) 
            self.num_trains = 0

    def train_step(self, rb):    
        b_data = random.sample(rb, self.batch_size)
        b_states = [data['states'] for data in b_data] 
        b_actions = torch.cat([data['actions'] for data in b_data]).to(self.device) 
        b_rewards = torch.cat([data['rewards'] for data in b_data]).to(self.device) 
        b_seq_lenghts = [len(data['states']) for data in b_data]
        b_next_dones = torch.cat([torch.tensor([0] * (b_seq_lenghts[i] - 2) + [1], dtype=torch.float32) for i in range(self.batch_size)]).to(self.device)
        init_gru_states = torch.zeros(self.actor.gru.num_layers, self.batch_size, self.actor.gru.hidden_size).to(self.device)

        # CRITIC training
        _, _, all_log_pi, all_action_probs, entropy = self.actor.get_actions(
            b_states,
            init_gru_states,
            b_seq_lenghts,
            self.device
        )
        next_state_action_probs = torch.cat([x[1:] for x in all_action_probs])
        next_state_log_pi = torch.cat([x[1:] for x in all_log_pi])

        with torch.no_grad():
            all_qf1_target, _ = self.qf1_target.get_values(
                b_states,
                init_gru_states,
                b_seq_lenghts,
                self.device
            )
            all_qf2_target, _ = self.qf2_target.get_values(
                b_states,
                init_gru_states,
                b_seq_lenghts,
                self.device
            )
            qf1_next_target = torch.cat([x[1:] for x in all_qf1_target])
            qf2_next_target = torch.cat([x[1:] for x in all_qf2_target])
    
            min_qf_next_target = next_state_action_probs * (
                torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            )

            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(1)
            next_q_value = b_rewards.reshape(-1, 1) + (1 - b_next_dones.reshape(-1, 1)) * self.gamma * (min_qf_next_target)

        all_qf1_values, _ = self.qf1.get_values(
            b_states,
            init_gru_states,
            b_seq_lenghts,
            self.device
        )
        all_qf2_values, _ = self.qf2.get_values(
            b_states,
            init_gru_states,
            b_seq_lenghts,
            self.device
        )
        qf1_values = torch.cat([x[:-1] for x in all_qf1_values])
        qf2_values = torch.cat([x[:-1] for x in all_qf2_values])
        qf1_a_values = qf1_values.gather(1, b_actions.reshape(-1, 1))
        qf2_a_values = qf2_values.gather(1, b_actions.reshape(-1, 1))
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # ACTOR training
        action_probs =  torch.cat([x[:-1] for x in all_action_probs])
        log_pi = torch.cat([x[:-1] for x in all_log_pi])

        with torch.no_grad():
            all_qf1_values, _ = self.qf1.get_values(
                b_states,
                init_gru_states,
                b_seq_lenghts,
                self.device
            )
            all_qf2_values, _ = self.qf2.get_values(
                b_states,
                init_gru_states,
                b_seq_lenghts,
                self.device
            )
            qf1_values = torch.cat([x[:-1] for x in all_qf1_values])
            qf2_values = torch.cat([x[:-1] for x in all_qf2_values])
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
            self.writer.add_scalar("losses/entropy", entropy.mean().item(), self.num_trains)
            if self.autotune:
                self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), self.num_trains)
                
            self.num_trains += 1

    def train(self, rb):
        if len(rb) < self.batch_size:
            raise Exception('Not enough samples in replay buffer')
        
        for _ in range(self.train_steps):
            self.train_step(rb)


        


        