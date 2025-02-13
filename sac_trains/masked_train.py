# from pettingzoo.classic import leduc_holdem_v4
from pettingzoo.classic import texas_holdem_v4
from algs.masked_sac.sac import SACAgentMLP
from algs.masked_sac.buffer import ReplayBuffer
from algs.masked_sac.networks import Actor
import time
import torch

def copy_model(model, device):
    copy_model = Actor(model.state_dim, model.action_dim)
    copy_model.load_state_dict(model.state_dict())
    return copy_model.to(device)

STATE_DIM = 72  # 36
ACTION_DIM = 4 
BUFFER_SIZE = 1_000_000
SAVE_PATH = 'history/holdem/' + time.strftime('%Y%m%d_%H%M%S')
NUM_EPOCHS = 100_000_000
ROLLOUT_DEVICE = 'cpu'
TRAIN_DEVICE = 'cuda'
START_BUFFER_SIZE = 10_000
TRAIN_FREQ = 1_000

agent = SACAgentMLP(
    state_dim = STATE_DIM,
    action_dim = ACTION_DIM,
    policy_lr = 1e-3,
    q_lr = 1e-3,
    batch_size = 100,
    train_steps = 10,
    gamma = 0.99,
    tau = 0.995,
    alpha = 0.1,
    autotune = True,
    target_entropy_scale = 0.89,
    device = TRAIN_DEVICE,
    # log_dir = SAVE_PATH
)

rb = ReplayBuffer(
    BUFFER_SIZE,
    STATE_DIM,
    ACTION_DIM,
    TRAIN_DEVICE
)

# env = leduc_holdem_v4.env()
env = texas_holdem_v4.env()
start_time = time.time()
mean_reward, mean_steps = 0, 0
rollout_actor = copy_model(agent.actor, ROLLOUT_DEVICE)

for epoch in range(1, NUM_EPOCHS):
    env.reset(seed=42)
    terminated, truncated = False, False
    states, action_masks, actions, rewards = [], [], [], []
    for player in env.agent_iter():
        state, reward, terminated, truncated, _ = env.last()

        if terminated or truncated:
            action = None
            if player == 'player_1':
                states.append(state['observation'])
                action_masks.append(state["action_mask"])
                rewards.append(reward)
        else:
            mask = state["action_mask"]
            if player == 'player_0':
                action = env.action_space(player).sample(mask)
            elif player == 'player_1':
                action, _, _ = rollout_actor.get_action(
                    torch.Tensor(state['observation']).to(ROLLOUT_DEVICE),
                    torch.Tensor(mask).to(ROLLOUT_DEVICE)
                )
                action = action.item()
                states.append(state['observation'])
                action_masks.append(mask)
                actions.append(action)
                rewards.append(reward)

        env.step(action)

    dones = [False] * (len(states) - 1) + [True]
    for i in range(len(states) - 1):
        rb.add(
            states[i],
            action_masks[i],
            actions[i],
            rewards[i + 1],
            states[i + 1],
            action_masks[i + 1],
            dones[i + 1]
        )

    ep_reward = sum(rewards)
    ep_steps = len(states)

    if agent.log_dir:
        agent.writer.add_scalar("rollout/ep_reward", ep_reward, epoch)
        agent.writer.add_scalar("rollout/ep_steps", ep_steps, epoch)

    mean_reward += ep_reward
    mean_steps += ep_steps

    if len(rb) > START_BUFFER_SIZE and epoch % TRAIN_FREQ == 0:
        print(f'epoch: {epoch} mean_reward: {round(mean_reward / TRAIN_FREQ, 2)} mean_steps: {round(mean_steps / TRAIN_FREQ, 2)} time: {round(time.time() - start_time, 2)} s')
        mean_reward, mean_steps = 0, 0
        if agent.log_dir:
            agent.writer.add_scalar("params/rb_length", len(rb), epoch)
        agent.train(rb)
        rollout_actor = copy_model(agent.actor, ROLLOUT_DEVICE)

    

