import gymnasium as gym
from algs.dqn.dqn import DQNAgent
from algs.dqn.utils import *
from torchrl.data import ReplayBuffer, LazyMemmapStorage
import time
import torch
import random

if __name__ == '__main__':
    STATE_DIM = 4
    ACTION_DIM = 2
    SAVE_PATH = 'train/history/cart_pole/' + time.strftime('%Y%m%d_%H%M%S')
    ENV_ID = 'CartPole-v1'
    ROLLOUT_DEVICE = 'cpu'
    TRAIN_DEVICE = 'cuda'
    RB_SIZE = 1_000_000
    LOG_INTERVAL = 100
    START_BUFFER_SIZE = 10_000
    TRAIN_FREQ = 100

    agent = DQNAgent(
        state_dim = STATE_DIM,
        action_dim = ACTION_DIM,
        lr = 1e-3,
        gamma = 0.99,
        tau = 0.995,
        batch_size = 256,
        train_steps = 100,
        device = TRAIN_DEVICE,
        # log_dir = SAVE_PATH
    )

    rb = ReplayBuffer(
        storage=LazyMemmapStorage(RB_SIZE)
    )

    env = gym.make(ENV_ID)
    start_time = time.time()
    mean_reward, mean_steps = 0, 0
    rollout_model = copy_model(agent.model, ROLLOUT_DEVICE)

    for epoch in range(1, int(1e10)):
        state, _ = env.reset()
        terminated, truncated = False, False
        ep_reward, ep_steps = 0, 0
        while not (terminated or truncated):
            action = rollout_model.get_action(
                torch.tensor(state).to(ROLLOUT_DEVICE)
            )
            action = action.item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            data = {
                'state': torch.tensor(state),
                'action': torch.tensor(action),
                'reward': torch.tensor(reward),
                'next_state': torch.tensor(next_state),
                'done': torch.tensor(terminated or truncated, dtype=torch.float32)
            }
            rb.add(data)
            state = next_state
            ep_reward += reward
            ep_steps += 1

        if agent.log_dir:
            agent.writer.add_scalar("rollout/ep_reward", ep_reward, epoch)
            agent.writer.add_scalar("rollout/ep_steps", ep_steps, epoch)

        mean_reward += ep_reward
        mean_steps += ep_steps

        if epoch % LOG_INTERVAL == 0:
            print(f'epoch: {epoch} mean_reward: {round(mean_reward / LOG_INTERVAL, 2)} mean_steps: {round(mean_steps / LOG_INTERVAL, 2)} buffer_size: {len(rb)} time: {round(time.time() - start_time, 2)} s')
            mean_reward, mean_steps = 0, 0

        if len(rb) > START_BUFFER_SIZE and epoch % TRAIN_FREQ == 0:
            agent.train(rb)
            rollout_model = copy_model(agent.model, ROLLOUT_DEVICE)

    

