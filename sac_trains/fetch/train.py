import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

# from algs.sac.sac import SACAgentMLP
from algs.sac.sac_continuous import SACAgentMLP
from algs.her import HER
from torchrl.data import ReplayBuffer, PrioritizedReplayBuffer, LazyMemmapStorage
import time
import torch
from copy import deepcopy

if __name__ == '__main__':
    ENV_ID = 'FetchReachDense-v4' 
    env = gym.make(ENV_ID)
    env = gym.wrappers.ClipAction(env)
    STATE_DIM = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
    ACTION_DIM = env.action_space.shape[0]
    SAVE_PATH = f'sac_trains/fetch/history/{ENV_ID}/' + time.strftime('%Y%m%d_%H%M%S')
    ROLLOUT_DEVICE = 'cpu'
    TRAIN_DEVICE = 'cuda'
    RB_SIZE = 1_000_000
    START_BUFFER_SIZE = 10_000
    TRAIN_FREQ = 5000

    agent = SACAgentMLP(
        state_dim = STATE_DIM,
        action_dim = ACTION_DIM,
        policy_lr = 5e-4,
        q_lr = 5e-4,
        a_lr = 1e-3,
        batch_size = 500,
        train_steps = 25,
        gamma = 0.99,
        tau = 0.005,
        alpha = 0.2,
        autotune = True,
        target_entropy_scale = 0.1,
        device = TRAIN_DEVICE,
        # log_dir = SAVE_PATH
    )

    rb = ReplayBuffer(
        storage=LazyMemmapStorage(RB_SIZE)
    )

    her = HER(
        prob = 0.2,
        k = 4,
        replay_buffer = rb
    )

    env = gym.make(ENV_ID)
    start_time = time.time()
    total_timesteps = 0
    reward_history, steps_history = [], []
    success_count = 0
    last_best_reward = None
    rollout_actor = deepcopy(agent.actor).to(ROLLOUT_DEVICE)

    for epoch in range(1, int(1e10)):
        state, _ = env.reset()
        obs, ach, des = torch.tensor(state['observation'], dtype=torch.float32), \
                        torch.tensor(state['achieved_goal'], dtype=torch.float32), \
                        torch.tensor(state['desired_goal'], dtype=torch.float32)
        terminated, truncated = False, False
        ep_reward, ep_steps = 0, 0
        while not (terminated or truncated):
            state = torch.cat((obs, des))
            with torch.no_grad():
                action, _, _ = rollout_actor.get_action(
                    state.to(ROLLOUT_DEVICE).unsqueeze(0)
                )
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            next_obs, next_ach, next_des = torch.tensor(next_state['observation'], dtype=torch.float32),   \
                                           torch.tensor(next_state['achieved_goal'], dtype=torch.float32), \
                                           torch.tensor(next_state['desired_goal'], dtype=torch.float32)
            data = {
                'state': state,
                'action': action.squeeze(0),
                'reward': torch.tensor(reward, dtype=torch.float32),
                'next_state': torch.cat((next_obs, next_des)),
                'next_done': torch.tensor(terminated or truncated, dtype=torch.float32)
            }
            rb.add(data)
            if her:
                her.add(obs, action.squeeze(0), next_obs, next_ach)
            obs, ach, des = next_obs, next_ach, next_des
            ep_reward += reward
            ep_steps += 1
            if terminated:
                success_count += 1

        total_timesteps += ep_steps 

        if agent.log_dir:
            agent.writer.add_scalar("rollout/mean_reward", ep_reward, total_timesteps)
            agent.writer.add_scalar("rollout/mean_steps", ep_steps, total_timesteps)

        reward_history += [ep_reward]
        steps_history += [ep_steps]    
    
        if sum(steps_history) > TRAIN_FREQ:
            mean_reward = sum(reward_history) / len(reward_history)
            mean_steps = sum(steps_history) / len(steps_history)
            success_rate = success_count / len(reward_history)

            print(f'epoch: {epoch} total_steps: {total_timesteps} mean_reward: {round(float(mean_reward), 2)} mean_steps: {round(float(mean_steps), 2)} success_rate: {round(success_rate, 2)}% buffer_size: {len(rb)} time: {round(time.time() - start_time, 2)}s')
            
            if last_best_reward is None:
                last_best_reward = mean_reward

            if mean_reward >= last_best_reward:
                if agent.log_dir:
                    torch.save(agent.actor.state_dict(), agent.log_dir + '/' + 'best_actor.pt') 
                last_best_reward = mean_reward
                print('New best model!')

            reward_history, steps_history = [], []
            success_count = 0

            if len(rb) > START_BUFFER_SIZE:
                agent.train(rb)
                rollout_actor = deepcopy(agent.actor).to(ROLLOUT_DEVICE)
