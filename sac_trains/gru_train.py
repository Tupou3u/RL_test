import gymnasium as gym
from algs.gru_sac.sac import SACAgentGRU
# from algs.gru_sac.sac_v1 import SACAgentGRU
# from algs.gru_sac.sac_s import SACAgentGRU
from torchrl.data import ReplayBuffer, LazyMemmapStorage
from algs.gru_sac.utils import *
from algs.gru_sac.sampler import RNNSampler
import time
import torch

if __name__ == '__main__':
    STATE_DIM = 2
    ACTION_DIM = 2
    SAVE_PATH = 'sac_trains/history/gru/cartpole/' + time.strftime('%Y%m%d_%H%M%S')
    ENV_ID = 'CartPole-v1'  # 'CartPole-v1', 'LunarLander-v3'
    ROLLOUT_DEVICE = 'cpu'
    TRAIN_DEVICE = 'cuda'
    RB_SIZE = 1_000_000
    START_BUFFER_SIZE = 10_000
    TRAIN_FREQ = 1000

    agent = SACAgentGRU(
        state_dim = STATE_DIM,
        action_dim = ACTION_DIM,
        policy_lr = 5e-4,
        q_lr = 5e-4,
        a_lr = 1e-3,
        batch_size = 100,
        num_batches = 5,
        train_steps = 25,
        gamma = 0.99,
        tau = 0.005,
        alpha = 0.01,
        autotune = True,
        target_entropy_scale = 0.89,
        device = TRAIN_DEVICE,
        log_dir = SAVE_PATH
    )

    rb = ReplayBuffer(
        storage=LazyMemmapStorage(RB_SIZE),
        sampler=RNNSampler()
    )

    env = gym.make(ENV_ID)
    start_time = time.time()
    total_steps = 0
    mean_reward, mean_steps = [], []
    rollout_actor = copy_model(agent.actor, ROLLOUT_DEVICE)

    for epoch in range(1, int(1e10)):
        state, _ = env.reset()
        state = state[[0, 2]]
        gru_state = torch.zeros(agent.actor.gru.num_layers, 1, agent.actor.gru.hidden_size).to(ROLLOUT_DEVICE)
        # gru_state = torch.zeros(agent.actor.shared_encoder.gru.num_layers, 1, agent.actor.shared_encoder.gru.hidden_size).to(ROLLOUT_DEVICE)
        start_flag, next_done = True, False
        ep_reward, ep_steps = 0, 0
        while not next_done:
            with torch.no_grad():
                action, next_gru_state, _, _, _ = rollout_actor.get_action(
                    torch.tensor(state).to(ROLLOUT_DEVICE),
                    gru_state,
                    torch.tensor(start_flag, dtype=torch.float32).to(ROLLOUT_DEVICE)
                )
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            next_state = next_state[[0, 2]]
            next_done = terminated or truncated
            data = {
                'state': torch.tensor(state, dtype=torch.float32),
                'gru_state': gru_state,
                'start_flag': torch.tensor(start_flag, dtype=torch.float32),
                'action': action,
                'reward': torch.tensor(reward, dtype=torch.float32),
                'next_state': torch.tensor(next_state, dtype=torch.float32),
                'next_done': torch.tensor(next_done, dtype=torch.float32)
            }
            rb.add(data)
            state = next_state
            gru_state = next_gru_state
            start_flag = False
            ep_reward += reward
            ep_steps += 1

        if agent.log_dir:
            agent.writer.add_scalar("rollout/ep_reward", ep_reward, epoch)
            agent.writer.add_scalar("rollout/ep_steps", ep_steps, epoch)

        mean_reward += [ep_reward]
        mean_steps += [ep_steps]     
        total_steps += ep_steps

        if sum(mean_steps) > TRAIN_FREQ:
            print(f'epoch: {epoch} total_steps: {total_steps} mean_reward: {round(sum(mean_reward) / len(mean_reward), 2)} mean_steps: {round(sum(mean_steps) / len(mean_steps), 2)} buffer_size: {len(rb)} time: {round(time.time() - start_time, 2)} s')
            mean_reward, mean_steps = [], []
            if len(rb) > START_BUFFER_SIZE:
                torch.autograd.set_detect_anomaly(True)
                agent.train(rb)
                rollout_actor = copy_model(agent.actor, ROLLOUT_DEVICE)

    

