# Векторизированное обучение

import gymnasium as gym
# from algs.ppo.ppo import PPO_MLP
from algs.ppo.ppo_continuous import PPO_MLP
import time
import torch
import numpy as np

def make_env(env_id):
    def thunk():
        env = gym.make(
            env_id, 
            # hardcore=True
            max_episode_steps=100
        )
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

if __name__ == '__main__':
    ENV_ID = 'MountainCarContinuous-v0'  # BipedalWalker-v3
    NUM_ENVS = 2
    envs = gym.vector.SyncVectorEnv(
        [make_env(ENV_ID) for i in range(NUM_ENVS)]
    )
    STATE_DIM = envs.single_observation_space.shape[0]
    ACTION_DIM = envs.single_action_space.shape[0]
    SAVE_PATH = f'ppo_trains/other/history/{ENV_ID}/' + time.strftime('%Y%m%d_%H%M%S')
    DEVICE = 'cuda'
    ROLLOUT_LEN = 500
    LOG_FREQ = 10
    
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
        device = DEVICE,
        # log_dir = SAVE_PATH
    ) 

    next_state, _ = envs.reset()
    next_done = [1] * NUM_ENVS
    start_time = time.time()
    total_timesteps = 0
    states = torch.zeros((ROLLOUT_LEN, NUM_ENVS, STATE_DIM)).to(DEVICE)
    actions = torch.zeros((ROLLOUT_LEN, NUM_ENVS, ACTION_DIM)).to(DEVICE)
    logprobs = torch.zeros((ROLLOUT_LEN, NUM_ENVS)).to(DEVICE)
    rewards = torch.zeros((ROLLOUT_LEN, NUM_ENVS)).to(DEVICE)
    next_dones = torch.zeros((ROLLOUT_LEN, NUM_ENVS)).to(DEVICE)
    values = torch.zeros((ROLLOUT_LEN, NUM_ENVS)).to(DEVICE)
    last_best_reward = None
    sum_rewards, num_dones = 0, 0

    for epoch in range(1, int(1e10)):
        for i in range(ROLLOUT_LEN):
            states[i] = torch.tensor(next_state).to(DEVICE)
            next_dones[i] = torch.tensor(next_done).to(DEVICE)

            with torch.no_grad():
                action, value, logprob, _ = agent.model.get_action_and_value(
                    torch.tensor(next_state).to(DEVICE)
                )
            next_state, reward, termination, truncation, info = envs.step(action.cpu().numpy())
            next_done = np.logical_or(termination, truncation)
            print(next_state.flatten(), next_done)

            # reward = np.zeros_like(termination, dtype=np.float32)
            # reward[termination == 1] = 1

            actions[i] = action
            values[i] = value
            rewards[i] = torch.tensor(reward).to(DEVICE)
            logprobs[i] = logprob

        total_timesteps += ROLLOUT_LEN * NUM_ENVS

        sum_rewards += torch.sum(rewards).item()
        num_dones += torch.sum(next_dones).item()

        if epoch % LOG_FREQ == 0:
            mean_reward = torch.sum(rewards).item() / torch.sum(next_dones).item()
            if agent.log_dir:
                agent.writer.add_scalar("rollout/mean_reward", mean_reward, epoch)
            sum_rewards, num_dones = 0, 0
            print(f'epoch: {epoch} total_timesteps: {total_timesteps} mean_reward: {round(mean_reward, 2)} time: {round(time.time() - start_time, 2)} s')

            if last_best_reward is None:
                last_best_reward = mean_reward

            if mean_reward >= last_best_reward:
                if agent.log_dir:
                    torch.save(agent.model.state_dict(), agent.log_dir + '/' + 'best_model.pt') 
                last_best_reward = mean_reward
                print('New best model!')
        
        agent.train(
            states[:-1], 
            actions[:-1], 
            rewards[:-1], 
            next_dones, 
            logprobs[:-1], 
            values, 
        )
    

