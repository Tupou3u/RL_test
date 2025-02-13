import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
import torch
# from algs.ppo.ppo import MLPAgent
from algs.ppo.ppo_continuous import MLPAgent
import time
import numpy as np

ENV_ID = 'FetchReach-v4'
NUM_EPS = 10
DEVICE = 'cpu'
env = gym.make(
    ENV_ID, 
    render_mode="human",
)
env = gym.wrappers.ClipAction(env)
STATE_DIM = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
ACTION_DIM = env.action_space.shape[0]
model = MLPAgent(STATE_DIM, ACTION_DIM).to(DEVICE)
model.load_state_dict(torch.load('ppo_trains/fetch/history/FetchReach-v4/20250211_231312/best_model.pt', weights_only=True))

mean_reward, mean_steps = [], []
success_rate = 0
for i in range(NUM_EPS):
    while True:
        next_state, _ = env.reset()
        if np.linalg.norm(next_state['desired_goal'] - next_state['achieved_goal']) > 0.05:
            break
    terminated, truncated = False, False
    ep_reward, ep_steps = 0, 0
    while not (terminated or truncated):
        with torch.no_grad():
            action = model.get_action(
                torch.cat((torch.tensor(next_state['observation'], dtype=torch.float32), torch.tensor(next_state['desired_goal'], dtype=torch.float32))).to(DEVICE).unsqueeze(0)
            )
        next_state, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
        if terminated:
            success_rate += 1

        ep_reward += reward
        ep_steps += 1
        env.render()
        time.sleep(0.1)
 
    mean_reward += [ep_reward]
    mean_steps += [ep_steps]

mean_reward = sum(mean_reward) / len(mean_reward)
mean_steps = sum(mean_steps) / len(mean_steps)
print(f'mean_reward: {round(mean_reward, 2)} mean_steps: {round(mean_steps, 2)} winrate: {round(100 * success_rate / NUM_EPS, 2)}%')
env.close()