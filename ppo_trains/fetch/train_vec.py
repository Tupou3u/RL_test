# Векторизированное обучение

import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

from algs.ppo.ppo_continuous import PPO_MLP
# from algs.ppo.ppo_rnad_continuous import PPO_MLP
import time
import torch
import numpy as np

def make_env(env_id):
    def thunk():
        env = gym.make(
            env_id, 
            # hardcore=True
        )
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

if __name__ == '__main__':
    ENV_ID = 'FetchReachDense-v4' 
    NUM_ENVS = 10
    envs = gym.vector.AsyncVectorEnv(
        [make_env(ENV_ID) for i in range(NUM_ENVS)]
    )
    STATE_DIM = envs.single_observation_space['observation'].shape[0] + envs.single_observation_space['desired_goal'].shape[0]
    ACTION_DIM = envs.single_action_space.shape[0]
    SAVE_PATH = f'ppo_trains/fetch/history/{ENV_ID}/' + time.strftime('%Y%m%d_%H%M%S')
    DEVICE = 'cuda'
    ROLLOUT_LEN = 500
    
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

    start_time = time.time()
    total_timesteps = 0
    states, _ = envs.reset()
    autoreset = np.zeros(NUM_ENVS)
    last_best_reward = None

    for epoch in range(1, int(1e10)):
        ep_data = {
            'states': [[] for _ in range(NUM_ENVS)],
            'actions': [[] for _ in range(NUM_ENVS)],
            'rewards': [[] for _ in range(NUM_ENVS)],
            'values': [[] for _ in range(NUM_ENVS)],
            'logprobs': [[] for _ in range(NUM_ENVS)],
            'next_dones': [[] for _ in range(NUM_ENVS)],
        }
        rollout_steps = [0] * NUM_ENVS
        success_rate = 0
        ep_rewards, ep_steps = [], []
        while any(i < ROLLOUT_LEN + 2 for i in rollout_steps):  
            states = torch.cat(
                (
                    torch.tensor(states['observation'], dtype=torch.float32), 
                    torch.tensor(states['desired_goal'], dtype=torch.float32)
                ), dim=1).to(DEVICE)    
            with torch.no_grad():
                actions, values, logprobs, _ = agent.model.get_action_and_value(states)
            next_states, rewards, terminations, truncations, infos = envs.step(actions.cpu().numpy())

            for j in range(envs.num_envs):
                if not autoreset[j]:
                    ep_data['states'][j].append(states[j])
                    ep_data['actions'][j].append(actions[j])
                    ep_data['rewards'][j].append(rewards[j])
                    ep_data['values'][j].append(values[j])
                    ep_data['logprobs'][j].append(logprobs[j])
                    ep_data['next_dones'][j].append(terminations[j] or truncations[j])
                    rollout_steps[j] += 1
                    if terminations[j]:
                        success_rate += 1
                if infos:
                    if '_episode' in infos:
                        if infos['_episode'][j]:
                            ep_rewards.append(infos['episode']['r'][j])
                            ep_steps.append(infos['episode']['l'][j])

                
            states = next_states
            autoreset = np.logical_or(terminations, truncations)

        total_timesteps += ROLLOUT_LEN * NUM_ENVS
        mean_reward = sum(ep_rewards) / len(ep_rewards)
        mean_steps = sum(ep_steps) / len(ep_steps)
        success_rate = success_rate / len(ep_rewards)

        if agent.log_dir:
            agent.writer.add_scalar("rollout/mean_reward", mean_reward, epoch)
            agent.writer.add_scalar("rollout/mean_steps", mean_steps, epoch)
            agent.writer.add_scalar("rollout/success_rate", success_rate, epoch)

        print(f'epoch: {epoch} total_timesteps: {total_timesteps} mean_reward: {round(float(mean_reward), 2)} mean_steps: {round(float(mean_steps), 2)} success_rate: {round(100 * success_rate, 2)}% time: {round(time.time() - start_time, 2)}s')

        if last_best_reward is None:
            last_best_reward = mean_reward

        if mean_reward >= last_best_reward:
            if agent.log_dir:
                torch.save(agent.model.state_dict(), agent.log_dir + '/' + 'best_model.pt') 
            last_best_reward = mean_reward
            print('New best model!')

        ep_data['states'] = [torch.stack(ep_data['states'][i][1:ROLLOUT_LEN+1]) for i in range(NUM_ENVS)]
        ep_data['actions'] = [torch.stack(ep_data['actions'][i][1:ROLLOUT_LEN+1]) for i in range(NUM_ENVS)]
        ep_data['rewards'] = [ep_data['rewards'][i][1:ROLLOUT_LEN+1] for i in range(NUM_ENVS)]
        ep_data['values'] = [ep_data['values'][i][1:ROLLOUT_LEN+2] for i in range(NUM_ENVS)]
        ep_data['logprobs'] = [ep_data['logprobs'][i][1:ROLLOUT_LEN+1] for i in range(NUM_ENVS)]
        ep_data['next_dones'] = [ep_data['next_dones'][i][:ROLLOUT_LEN+1] for i in range(NUM_ENVS)]
       
        agent.train(
            torch.stack(ep_data['states']).permute(1, 0, 2), 
            torch.stack(ep_data['actions']).to(DEVICE).permute(1, 0, 2), 
            torch.tensor(ep_data['rewards']).to(DEVICE).permute(1, 0), 
            torch.tensor(ep_data['next_dones'], dtype=torch.float32).to(DEVICE).permute(1, 0), 
            torch.tensor(ep_data['logprobs']).to(DEVICE).permute(1, 0), 
            torch.tensor(ep_data['values']).to(DEVICE).permute(1, 0) 
        )
    

