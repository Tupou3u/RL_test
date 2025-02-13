import gymnasium as gym
from algs.sac.sac import SACAgentMLP
from torchrl.data import ReplayBuffer, PrioritizedReplayBuffer, LazyMemmapStorage
import time
import torch
from copy import deepcopy

if __name__ == '__main__':
    ENV_ID = 'LunarLander-v3'
    env = gym.make(ENV_ID)
    # env = gym.wrappers.ClipAction(env)
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.n
    SAVE_PATH = f'sac_trains/other/history/{ENV_ID}/' + time.strftime('%Y%m%d_%H%M%S')
    ROLLOUT_DEVICE = 'cpu'
    TRAIN_DEVICE = 'cuda'
    RB_SIZE = 1_000_000
    START_BUFFER_SIZE = 10_000
    TRAIN_FREQ = 10000

    agent = SACAgentMLP(
        state_dim = STATE_DIM,
        action_dim = ACTION_DIM,
        policy_lr = 5e-4,
        q_lr = 5e-4,
        a_lr = 1e-3,
        batch_size = 500,
        train_steps = 20,
        gamma = 0.99,
        tau = 0.005,
        alpha = 0.2,
        autotune = True,
        target_entropy_scale = 0.01,
        device = TRAIN_DEVICE,
        log_dir = SAVE_PATH
    )

    rb = ReplayBuffer(
        storage=LazyMemmapStorage(RB_SIZE)
    )

    env = gym.make(ENV_ID)
    start_time = time.time()
    total_steps = 0
    mean_reward, mean_steps = [], []
    rollout_actor = deepcopy(agent.actor).to(ROLLOUT_DEVICE)

    for epoch in range(1, int(1e10)):
        state, _ = env.reset()
        terminated, truncated = False, False
        ep_reward, ep_steps = 0, 0
        while not (terminated or truncated):
            with torch.no_grad():
                action, _, _ = rollout_actor.get_action(
                    torch.tensor(state).to(ROLLOUT_DEVICE)
                )
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            data = {
                'state': torch.tensor(state, dtype=torch.float32),
                'action': action,
                'reward': torch.tensor(reward, dtype=torch.float32),
                'next_state': torch.tensor(next_state, dtype=torch.float32),
                'next_done': torch.tensor(terminated or truncated, dtype=torch.float32)
            }
            rb.add(data)
            state = next_state
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
                agent.train(rb)
                rollout_actor = deepcopy(agent.actor).to(ROLLOUT_DEVICE)

    

