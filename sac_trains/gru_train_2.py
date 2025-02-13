import gymnasium as gym
from algs.gru_sac_v2.sac import SACAgentGRU
from algs.gru_sac_v2.utils import *
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
    START_BUFFER_SIZE = 100
    TRAIN_FREQ = 5000

    agent = SACAgentGRU(
        state_dim = STATE_DIM,
        action_dim = ACTION_DIM,
        policy_lr = 5e-4,
        q_lr = 5e-4,
        a_lr = 1e-3,
        batch_size = 10,
        train_steps = 50,
        gamma = 0.99,
        tau = 0.005,
        alpha = 0.01,
        autotune = True,
        target_entropy_scale = 0.89,
        device = TRAIN_DEVICE,
        # log_dir = SAVE_PATH
    )

    env = gym.make(ENV_ID, max_episode_steps=100)
    start_time = time.time()
    total_steps = 0
    mean_reward, mean_steps = [], []
    rollout_actor = copy_model(agent.actor, ROLLOUT_DEVICE)
    rb = []

    for epoch in range(1, int(1e10)):
        state, _ = env.reset()
        state = state[[0, 2]]
        next_gru_state = torch.zeros(agent.actor.gru.num_layers, 1, agent.actor.gru.hidden_size).to(ROLLOUT_DEVICE)
        terminated, truncated = False, False
        ep_reward, ep_steps = 0, 0
        states, actions, rewards = [], [], []
        while not (terminated or truncated):
            with torch.no_grad():
                action, next_gru_state, _, _, _ = rollout_actor.get_action(
                    torch.tensor(state).to(ROLLOUT_DEVICE),
                    next_gru_state
                )
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            states.append(torch.tensor(state, dtype=torch.float32))
            actions.append(action)
            rewards.append(torch.tensor(reward, dtype=torch.float32))

            state = next_state
            state = next_state[[0, 2]]
            start_flag = False
            ep_reward += reward
            ep_steps += 1

        states.append(torch.tensor(state, dtype=torch.float32))
        data = {
            'states': torch.stack(states).to(TRAIN_DEVICE),
            'actions': torch.stack(actions).to(TRAIN_DEVICE),
            'rewards': torch.stack(rewards).to(TRAIN_DEVICE)
        }
        rb.append(data)
        rb[-RB_SIZE:]

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
                rollout_actor = copy_model(agent.actor, ROLLOUT_DEVICE)

    

