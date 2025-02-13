from pettingzoo.classic import leduc_holdem_v4
from algs.masked_sac.sac import SACAgentMLP
from algs.masked_sac.buffer import ReplayBuffer
from algs.masked_sac.networks import Actor

STATE_DIM = 36
ACTION_DIM = 4 
BUFFER_SIZE = 1_000_000
TRAIN_DEVICE = 'cpu'

rb = ReplayBuffer(
    BUFFER_SIZE,
    STATE_DIM,
    ACTION_DIM,
    TRAIN_DEVICE
)

env = leduc_holdem_v4.env()

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
            action = env.action_space(player).sample(mask)
            action = action.item()
            states.append(state['observation'])
            action_masks.append(mask)
            actions.append(action)
            rewards.append(reward)
    env.step(action)

print(len(states))

dones = [False] * (len(states) - 1) + [True]
for i in range(len(states) - 1):
    print(
        states[i],
        action_masks[i],
        actions[i],
        rewards[i + 1],
        states[i + 1],
        action_masks[i + 1],
        dones[i + 1]
    )
    rb.add(
        states[i],
        action_masks[i],
        actions[i],
        rewards[i + 1],
        states[i + 1],
        action_masks[i + 1],
        dones[i + 1]
    )

print(
    rb.states,
    rb.action_masks,
    rb.actions,
    rb.rewards,
    rb.next_states,
    rb.next_action_masks,
    rb.dones
)