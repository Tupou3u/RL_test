# Векторизированное обучение

import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

# from algs.sac.sac import SACAgentMLP
from algs.sac.sac_continuous import SAC_Continuous
from algs.her import HER
from torchrl.data import ReplayBuffer, PrioritizedReplayBuffer, LazyMemmapStorage
import time
import torch
import numpy as np

def make_env(env_id):
    def thunk():
        env = gym.make(
            env_id,
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
    envs = gym.vector.SyncVectorEnv(
        [make_env(ENV_ID) for i in range(NUM_ENVS)]
    )
    STATE_DIM = envs.single_observation_space['observation'].shape[0] + envs.single_observation_space['desired_goal'].shape[0]
    ACTION_DIM = envs.single_action_space.shape[0]
    SAVE_PATH = f'sac_trains/fetch/history/{ENV_ID}/' + time.strftime('%Y%m%d_%H%M%S')
    DEVICE = 'cuda'
    RB_SIZE = 1_000_000
    START_BUFFER_SIZE = 10_000
    ROLLOUT_LEN = 500
    
    agent = SAC_Continuous(
        state_dim = STATE_DIM,
        action_dim = ACTION_DIM,
        policy_lr = 5e-4,
        q_lr = 1e-3,
        a_lr = 1e-3,
        batch_size = 500,
        train_steps = 25,
        gamma = 0.99,
        tau = 0.005,
        alpha = 0.2,
        autotune = True,
        device = DEVICE,
        # log_dir = SAVE_PATH
    )

    rb = ReplayBuffer(
        storage=LazyMemmapStorage(RB_SIZE)
    )

    her = None
    # her = HER(
    #     prob = 0.5,
    #     k = 4,
    #     term_reward = 0.0,
    #     replay_buffer = rb
    # )

    start_time = time.time()
    total_timesteps = 0
    states, _ = envs.reset()
    obss, achs, dess = torch.tensor(states['observation'], dtype=torch.float32),   \
                       torch.tensor(states['achieved_goal'], dtype=torch.float32), \
                       torch.tensor(states['desired_goal'], dtype=torch.float32)
    autoreset = np.zeros(NUM_ENVS)
    last_best_reward = None

    for epoch in range(1, int(1e10)):
        rollout_steps, success_count = 0, 0
        ep_rewards, ep_steps = [], []
        while rollout_steps < ROLLOUT_LEN * NUM_ENVS:
            states = torch.cat((obss, dess), dim=1).to(DEVICE)
            with torch.no_grad():
                actions, _, _ = agent.actor.get_action(states)
            next_states, rewards, terminations, truncations, infos = envs.step(actions.cpu().numpy())
            next_obss, next_achs, next_dess = torch.tensor(next_states['observation'], dtype=torch.float32),   \
                                              torch.tensor(next_states['achieved_goal'], dtype=torch.float32), \
                                              torch.tensor(next_states['desired_goal'], dtype=torch.float32)

            for j in range(envs.num_envs):
                if not autoreset[j]:
                    data = {
                        'state': states[j],
                        'action': actions[j],
                        'reward': torch.tensor(rewards[j], dtype=torch.float32),
                        'next_state': torch.cat((next_obss[j], next_dess[j])),
                        'next_done': torch.tensor(terminations[j] or truncations[j], dtype=torch.float32)
                    }
                    rb.add(data)
                    if her:
                        her.add(obss[j], actions[j], next_obss[j], next_achs[j])
                    if terminations[j]:
                        success_count += 1
                    rollout_steps += 1
                if infos:
                    if '_episode' in infos:
                        if infos['_episode'][j]:
                            ep_rewards.append(infos['episode']['r'][j])
                            ep_steps.append(infos['episode']['l'][j])

            obss, achs, dess = next_obss, next_achs, next_dess
            autoreset = np.logical_or(terminations, truncations)

        total_timesteps += ROLLOUT_LEN * NUM_ENVS

        mean_reward = sum(ep_rewards) / len(ep_rewards)
        mean_steps = sum(ep_steps) / len(ep_steps)
        success_rate = success_count / len(ep_rewards)

        print(f'epoch: {epoch} total_timesteps: {total_timesteps} mean_reward: {round(float(mean_reward), 2)} mean_steps: {round(float(mean_steps), 2)} success_rate: {round(100 * success_rate, 2)}% buffer_size: {len(rb)} time: {round(time.time() - start_time, 2)}s')

        if last_best_reward is None:
            last_best_reward = mean_reward

        if mean_reward >= last_best_reward:
            if agent.log_dir:
                torch.save(agent.actor.state_dict(), agent.log_dir + '/' + 'best_actor.pt') 
            last_best_reward = mean_reward
            print('New best model!')
        
        if agent.log_dir:
            agent.writer.add_scalar("rollout/mean_reward", mean_reward, total_timesteps)
            agent.writer.add_scalar("rollout/mean_steps", mean_steps, total_timesteps)  

        if len(rb) > START_BUFFER_SIZE:
            agent.train(rb)
    

