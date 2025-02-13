# Однопоточное обучение

import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

# from algs.ppo.ppo import PPO_MLP
from algs.ppo.ppo_continuous import PPO_MLP
import time
import torch
import numpy as np
from copy import deepcopy

if __name__ == '__main__':
    ENV_ID = 'FetchReach-v4' 
    env = gym.make(ENV_ID)
    env = gym.wrappers.ClipAction(env)
    STATE_DIM = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
    ACTION_DIM = env.action_space.shape[0]
    SAVE_PATH = f'ppo_trains/fetch/history/{ENV_ID}/' + time.strftime('%Y%m%d_%H%M%S')
    ROLLOUT_DEVICE = 'cpu'
    TRAIN_DEVICE = 'cuda'
    ROLLOUT_LEN = 5000

    agent = PPO_MLP(
        state_dim = STATE_DIM,
        action_dim = ACTION_DIM,
        learning_rate = 5e-4,             
        batch_size = 1000,
        train_epochs = 10,
        gamma = 0.99,
        gae_lambda = 0.95,
        actor_clip = 0.1,
        critic_clip = 0.1,
        ent_coef = 0.001,      
        alpha = 0.1,                    
        device = TRAIN_DEVICE,
        log_dir = SAVE_PATH
    ) 

    start_time = time.time()
    total_timesteps = 0
    last_best_reward = None

    for epoch in range(1, int(1e10)):
        states, actions, rewards, logprobs, values, next_dones = [], [], [], [], [], []
        rollout_model = deepcopy(agent.model).to(ROLLOUT_DEVICE)
        mean_reward, mean_steps, success_count = [], [], 0

        while len(states) < ROLLOUT_LEN:
            state, _ = env.reset()
            next_done = False
            ep_reward, ep_steps = 0, 0
            while not next_done:
                state = torch.cat(
                    (
                        torch.tensor(state['observation'], dtype=torch.float32), 
                        torch.tensor(state['desired_goal'], dtype=torch.float32)
                    )
                )
                with torch.no_grad():
                    action, value, logprob, _ = rollout_model.get_action_and_value(
                        state.to(ROLLOUT_DEVICE).unsqueeze(0)
                    )
                next_state, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
                next_done = terminated or truncated
                if terminated:
                    success_count += 1

                states.append(state)
                actions.append(action)
                values.append(value)
                rewards.append(reward)
                logprobs.append(logprob)
                next_dones.append(next_done)

                state = next_state
                ep_reward += reward
                ep_steps += 1

            mean_reward += [ep_reward]
            mean_steps += [ep_steps]

        total_timesteps += sum(mean_steps)
        num_eps = len(mean_reward)
        mean_reward = sum(mean_reward) / num_eps
        mean_steps = sum(mean_steps) / num_eps
        success_rate = success_count / num_eps

        if last_best_reward is None:
            last_best_reward = mean_reward

        if agent.log_dir:
            agent.writer.add_scalar("rollout/mean_reward", mean_reward, epoch)
            agent.writer.add_scalar("rollout/mean_steps", mean_steps, epoch)
            agent.writer.add_scalar("rollout/success_rate", success_rate, epoch)

        print(f'epoch: {epoch} total_timesteps: {total_timesteps} mean_reward: {round(float(mean_reward), 2)} mean_steps: {round(float(mean_steps), 2)} winrate: {round(100 * success_rate, 2)}% time: {round(time.time() - start_time, 2)} s')

        if mean_reward >= last_best_reward:
            if agent.log_dir:
                torch.save(agent.model.state_dict(), agent.log_dir + '/' + 'best_model.pt') 
            last_best_reward = mean_reward
            print('New best model!')
        
        agent.train(
            torch.stack(states[:ROLLOUT_LEN-1]).to(TRAIN_DEVICE), 
            torch.stack(actions[:ROLLOUT_LEN-1]).to(TRAIN_DEVICE), 
            torch.tensor(rewards[:ROLLOUT_LEN-1], dtype=torch.float32).to(TRAIN_DEVICE), 
            torch.tensor([True] + next_dones[:ROLLOUT_LEN-1], dtype=torch.float32).to(TRAIN_DEVICE), 
            torch.tensor(logprobs[:ROLLOUT_LEN-1], dtype=torch.float32).to(TRAIN_DEVICE), 
            torch.tensor(values[:ROLLOUT_LEN], dtype=torch.float32).to(TRAIN_DEVICE), 
        )
    

