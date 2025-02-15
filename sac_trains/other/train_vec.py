# Векторизированное обучение

import gymnasium as gym
# from algs.sac.sac import SACAgentMLP
from algs.sac.sac_continuous import SAC_Continuous
from torchrl.data import ReplayBuffer, PrioritizedReplayBuffer, LazyMemmapStorage
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
    ENV_ID = 'BipedalWalker-v3'  # 'BipedalWalker-v3'
    NUM_ENVS = 10
    envs = gym.vector.SyncVectorEnv(
        [make_env(ENV_ID) for _ in range(NUM_ENVS)]
    )
    STATE_DIM = envs.single_observation_space.shape[0]
    ACTION_DIM = envs.single_action_space.shape[0]
    SAVE_PATH = f'sac_trains/other/history/{ENV_ID}/' + time.strftime('%Y%m%d_%H%M%S')
    DEVICE = 'cuda'
    RB_SIZE = 1_000_000
    START_BUFFER_SIZE = 10_000
    ROLLOUT_LEN = 1000
    
    # agent = SACAgentMLP(
    #     state_dim = STATE_DIM,
    #     action_dim = ACTION_DIM,
    #     policy_lr = 5e-4,
    #     q_lr = 5e-4,
    #     a_lr = 1e-3,
    #     batch_size = 500,
    #     train_steps = 25,
    #     gamma = 0.99,
    #     tau = 0.005,
    #     alpha = 0.2,
    #     autotune = True,
    #     target_entropy_scale = 0.01,
    #     device = DEVICE,
    #     log_dir = SAVE_PATH
    # )

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
        log_dir = SAVE_PATH
    )

    rb = ReplayBuffer(
        storage=LazyMemmapStorage(RB_SIZE)
    )

    start_time = time.time()
    total_timesteps = 0
    states, _ = envs.reset()
    autoreset = np.zeros(NUM_ENVS)
    last_best_reward = None

    for epoch in range(1, int(1e10)):
        rollout_steps = 0
        ep_rewards, ep_steps = [], []
        while rollout_steps < ROLLOUT_LEN * NUM_ENVS:
            states = torch.tensor(states, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                actions, _, _ = agent.actor.get_action(states)
            next_states, rewards, terminations, truncations, infos = envs.step(actions.cpu().numpy())

            for j in range(envs.num_envs):
                if not autoreset[j]:
                    data = {
                        'state': states[j],
                        'action': actions[j],
                        'reward': torch.tensor(rewards[j], dtype=torch.float32),
                        'next_state': torch.tensor(next_states[j], dtype=torch.float32),
                        'next_done': torch.tensor(terminations[j] or truncations[j], dtype=torch.float32)
                    }
                    rb.add(data)
                    rollout_steps += 1
                if infos:
                    if '_episode' in infos:
                        if infos['_episode'][j]:
                            ep_rewards.append(infos['episode']['r'][j])
                            ep_steps.append(infos['episode']['l'][j])

            states = next_states
            autoreset = np.logical_or(terminations, truncations)

        total_timesteps += ROLLOUT_LEN * NUM_ENVS

        if ep_rewards:
            mean_reward = sum(ep_rewards) / len(ep_rewards)
            mean_steps = sum(ep_steps) / len(ep_steps)

            print(f'epoch: {epoch} total_timesteps: {total_timesteps} mean_reward: {round(float(mean_reward), 2)} mean_steps: {round(float(mean_steps), 2)} buffer_size: {len(rb)} time: {round(time.time() - start_time, 2)} s')

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
    

