import gymnasium as gym
import torch
# from algs.ppo.ppo import MLPAgent
from algs.ppo.ppo_continuous import MLPAgent

ENV_ID = 'BipedalWalker-v3'  # BipedalWalker-v3, Pendulum-v1
NUM_EPS = 10
DEVICE = 'cpu'
env = gym.make(
    ENV_ID, 
    hardcore=True, 
    render_mode="human"
)
env = gym.wrappers.ClipAction(env)
model = MLPAgent(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
model.load_state_dict(torch.load('ppo_trains/other/history/BipedalWalker-v3_hard/20250209_193620/best_model.pt', weights_only=True))

mean_reward, mean_steps = [], []
for i in range(NUM_EPS):
    state, _ = env.reset()
    terminated, truncated = False, False
    ep_reward, ep_steps = 0, 0
    while not (terminated or truncated):
        with torch.no_grad():
            action = model.get_action(
                torch.tensor(state).to(DEVICE).unsqueeze(0)
            )
        next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
        env.render()
        next_done = terminated or truncated
        state = next_state
        ep_reward += reward
        ep_steps += 1

    mean_reward += [ep_reward]
    mean_steps += [ep_steps]

mean_reward = sum(mean_reward) / len(mean_reward)
mean_steps = sum(mean_steps) / len(mean_steps)
print(f'mean_reward: {round(mean_reward, 2)} mean_steps: {round(mean_steps, 2)}')
env.close()