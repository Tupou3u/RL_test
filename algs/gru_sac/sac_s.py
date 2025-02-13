import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from algs.gru_sac.networks_shared import *
from torchrl.data import ReplayBuffer
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy

class SACAgentGRU:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            policy_lr: float = 1e-4,
            q_lr: float = 1e-4,
            a_lr: float = 1e-3,
            batch_size: int = 100,
            num_batches: int = 10,
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
        self.num_batches = num_batches
        self.train_steps = train_steps
        self.gamma = gamma
        self.tau = tau
        self.autotune = autotune
        self.device = device
        self.log_dir = log_dir

        self.shared_encoder = GRUEncoder(state_dim).to(device)
        self.actor = GRUActor(self.shared_encoder, action_dim).to(device)
        self.qf1 = GRUSoftQNetwork(self.shared_encoder, action_dim).to(device)
        self.qf2 = GRUSoftQNetwork(self.shared_encoder, action_dim).to(device)
        self.qf1_target = GRUSoftQNetwork(deepcopy(self.shared_encoder), action_dim).to(device)
        self.qf2_target = GRUSoftQNetwork(deepcopy(self.shared_encoder), action_dim).to(device)
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

    def train_step(self, replayBuffer: ReplayBuffer):    
        seq_len = self.batch_size * self.num_batches
        b_data = replayBuffer.sample(seq_len)
        b_states = b_data['state'].to(self.device) 
        b_gru_states = b_data['gru_state'].to(self.device)
        b_start_flags = b_data['start_flag'].to(self.device)
        b_actions = b_data['action'].to(self.device)
        b_rewards = b_data['reward'].to(self.device)
        b_next_states = b_data['next_state'].to(self.device)
        b_next_dones = b_data['next_done'].to(self.device)
        
        all_states, all_start_flags, all_dones = [], [], []
        t0 = torch.tensor(0, dtype=torch.float32).to(self.device)
        t1 = torch.tensor(1, dtype=torch.float32).to(self.device)
        for start in range(0, seq_len, self.batch_size):
            end = start + self.batch_size
            mb_all_states, mb_all_start_flags, mb_all_dones = [b_states[start], b_next_states[start]], [b_start_flags[start], t0], [t0, b_next_dones[start]]
            for state, next_state, next_done in zip(b_states[start + 1:end], b_next_states[start + 1:end], b_next_dones[start + 1:end]):
                if any(state != mb_all_states[-1]):
                    mb_all_states += [state, next_state]
                    mb_all_start_flags += [t1, t0]
                    mb_all_dones += [t0, next_done]
                else:
                    mb_all_states += [next_state]
                    mb_all_start_flags += [t0]
                    mb_all_dones += [next_done]

            all_states.append(torch.stack(mb_all_states))
            all_start_flags.append(torch.stack(mb_all_start_flags))
            all_dones.append(torch.stack(mb_all_dones))

        all_states = pad_sequence(all_states, batch_first=True, padding_value=0).permute(1, 0, 2)
        all_start_flags = pad_sequence(all_start_flags, batch_first=True, padding_value=-1)
        all_dones = pad_sequence(all_dones, batch_first=True, padding_value=-1)

        selected_states = b_gru_states[::self.batch_size]
        gru_states = torch.cat([state for state in selected_states], dim=1)

        # CRITIC training
        _, _, all_log_pi, all_action_probs, entropy = self.actor.get_action(
            all_states,
            gru_states,
            all_start_flags
        )
        next_state_inds = [torch.where(i[1:] == 0)[0] + 1 for i in all_start_flags] 
        next_state_log_pi = torch.cat([all_log_pi[inds, row] for row, inds in enumerate(next_state_inds)])
        next_state_action_probs = torch.cat([all_action_probs[inds, row] for row, inds in enumerate(next_state_inds)])

        with torch.no_grad():
            all_qf1_target, _ = self.qf1_target.get_value(
                all_states,
                gru_states,
                all_start_flags
            )
            all_qf2_target, _ = self.qf2_target.get_value(
                all_states,
                gru_states,
                all_start_flags
            )
            qf1_next_target = torch.cat([all_qf1_target[inds, row] for row, inds in enumerate(next_state_inds)])
            qf2_next_target = torch.cat([all_qf2_target[inds, row] for row, inds in enumerate(next_state_inds)])
    
            min_qf_next_target = next_state_action_probs * (
                torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            )
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(1)
            next_q_value = b_rewards.reshape(-1, 1) + (1 - b_next_dones.reshape(-1, 1)) * self.gamma * (min_qf_next_target)

        all_qf1_values, _ = self.qf1.get_value(
            all_states,
            gru_states,
            all_start_flags
        )
        all_qf2_values, _ = self.qf2.get_value(
            all_states,
            gru_states,
            all_start_flags
        )
        state_inds = [torch.where(i[:-1] == 0)[0] for i in all_dones] 
        qf1_values = torch.cat([all_qf1_values[inds, row] for row, inds in enumerate(state_inds)])
        qf2_values = torch.cat([all_qf2_values[inds, row] for row, inds in enumerate(state_inds)])

        qf1_a_values = qf1_values.gather(1, b_actions.reshape(-1, 1))
        qf2_a_values = qf2_values.gather(1, b_actions.reshape(-1, 1))
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # ACTOR training
        action_probs =  torch.cat([all_action_probs[inds, row] for row, inds in enumerate(state_inds)])
        log_pi =  torch.cat([all_log_pi[inds, row] for row, inds in enumerate(state_inds)])

        with torch.no_grad():
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

    def train(self, replayBuffer: ReplayBuffer):
        if len(replayBuffer) < self.batch_size:
            raise Exception('Not enough samples in replay buffer')
        
        for _ in range(self.train_steps):
            self.train_step(replayBuffer)


        


        