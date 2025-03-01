import gymnasium as gym
# from algs.sac.sac import SACAgentMLP
from algs.sac.sac_continuous import SACAgentMLP
from torchrl.data import ReplayBuffer, PrioritizedReplayBuffer, LazyMemmapStorage
import time
import torch
from copy import deepcopy

if __name__ == '__main__':
    ENV_ID = 'Pendulum-v1'
    env = gym.make(ENV_ID)
    env = gym.wrappers.ClipAction(env)
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]
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

    env = gym.make(ENV_ID)
    start_time = time.time()
    total_timesteps = 0
    reward_history, steps_history = [], []
    last_best_reward = None
    rollout_actor = deepcopy(agent.actor).to(ROLLOUT_DEVICE)

    for epoch in range(1, int(1e10)):
        state, _ = env.reset()
        terminated, truncated = False, False
        ep_reward, ep_steps = 0, 0
        while not (terminated or truncated):
            state = torch.tensor(state)
            with torch.no_grad():
                action, _, _ = rollout_actor.get_action(
                    state.to(ROLLOUT_DEVICE).unsqueeze(0)
                )
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            data = {
                'state': state,
                'action': action.squeeze(0),
                'reward': torch.tensor(reward, dtype=torch.float32),
                'next_state': torch.tensor(next_state, dtype=torch.float32),
                'next_done': torch.tensor(terminated or truncated, dtype=torch.float32)
            }
            rb.add(data)
            state = next_state
            ep_reward += reward
            ep_steps += 1

        total_timesteps += ep_steps 

        if agent.log_dir:
            agent.writer.add_scalar("rollout/mean_reward", ep_reward, total_timesteps)
            agent.writer.add_scalar("rollout/mean_steps", ep_steps, total_timesteps)

        reward_history += [ep_reward]
        steps_history += [ep_steps]    
    
        if sum(steps_history) > TRAIN_FREQ:
            mean_reward = sum(reward_history) / len(reward_history)
            mean_steps = sum(steps_history) / len(steps_history)

            print(f'epoch: {epoch} total_steps: {total_timesteps} mean_reward: {round(mean_reward, 2)} mean_steps: {round(mean_steps, 2)} buffer_size: {len(rb)} time: {round(time.time() - start_time, 2)} s')
            
            if last_best_reward is None:
                last_best_reward = mean_reward

            if mean_reward >= last_best_reward:
                if agent.log_dir:
                    torch.save(agent.actor.state_dict(), agent.log_dir + '/' + 'best_actor.pt') 
                last_best_reward = mean_reward
                print('New best model!')

            reward_history, steps_history = [], []

            if len(rb) > START_BUFFER_SIZE:
                agent.train(rb)
                rollout_actor = deepcopy(agent.actor).to(ROLLOUT_DEVICE)

    

