# Однопоточное обучение

import gymnasium as gym
# from algs.ppo.ppo import PPO_MLP
from algs.ppo.ppo_continuous import PPO_MLP
import time
import torch
from copy import deepcopy

if __name__ == '__main__':
    ENV_ID = 'Pendulum-v1'
    env = gym.make(ENV_ID)
    env = gym.wrappers.ClipAction(env)
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]
    SAVE_PATH = f'ppo_trains/history/{ENV_ID}/' + time.strftime('%Y%m%d_%H%M%S')
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
        ent_coef = 0.01,      
        alpha = 0.1,                    
        device = TRAIN_DEVICE,
        # log_dir = SAVE_PATH
    ) 

    start_time = time.time()
    total_timesteps = 0
    last_best_reward = None

    for epoch in range(1, int(1e10)):
        states, actions, rewards, logprobs, values, next_dones = [], [], [], [], [], []
        rollout_model = deepcopy(agent.model).to(ROLLOUT_DEVICE)
        mean_reward, mean_steps = [], []

        while len(states) < ROLLOUT_LEN + 1:
            state, _ = env.reset()
            terminated, truncated = False, False
            ep_reward, ep_steps = 0, 0
            while not (terminated or truncated):
                state = torch.tensor(state)
                with torch.no_grad():
                    action, value, logprob, _ = rollout_model.get_action_and_value(
                        state.to(ROLLOUT_DEVICE).unsqueeze(0)
                    )
                next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
                next_done = terminated or truncated

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
        mean_reward = sum(mean_reward) / len(mean_reward)
        mean_steps = sum(mean_steps) / len(mean_steps)

        if agent.log_dir:
            agent.writer.add_scalar("rollout/mean_reward", ep_reward, epoch)
            agent.writer.add_scalar("rollout/mean_steps", ep_steps, epoch)

        print(f'epoch: {epoch} total_timesteps: {total_timesteps} mean_reward: {round(mean_reward, 2)} mean_steps: {round(mean_steps, 2)} time: {round(time.time() - start_time, 2)} s')

        if last_best_reward is None:
            last_best_reward = mean_reward

        if mean_reward >= last_best_reward:
            if agent.log_dir:
                torch.save(agent.model.state_dict(), agent.log_dir + '/' + 'best_model.pt') 
            last_best_reward = mean_reward
            print('New best model!')
        
        agent.train(
            torch.stack(states[:ROLLOUT_LEN]).to(TRAIN_DEVICE), 
            torch.stack(actions[:ROLLOUT_LEN]).to(TRAIN_DEVICE), 
            torch.tensor(rewards[:ROLLOUT_LEN], dtype=torch.float32).to(TRAIN_DEVICE), 
            torch.tensor([True] + next_dones[:ROLLOUT_LEN], dtype=torch.float32).to(TRAIN_DEVICE), 
            torch.tensor(logprobs[:ROLLOUT_LEN], dtype=torch.float32).to(TRAIN_DEVICE), 
            torch.tensor(values[:ROLLOUT_LEN+1], dtype=torch.float32).to(TRAIN_DEVICE), 
        )
    

