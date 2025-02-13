import gymnasium as gym
from algs.sac.per_sac import PER_SACAgentMLP
# from torchrl.data import PrioritizedReplayBuffer, LazyMemmapStorage
from algs.sac.per import PrioritizedReplayBuffer 
from algs.sac.utils import *
import time
import torch

if __name__ == '__main__':
    STATE_DIM = 8
    ACTION_DIM = 4
    SAVE_PATH = 'train/history/lunar/' + time.strftime('%Y%m%d_%H%M%S')
    ENV_ID = 'LunarLander-v2'
    ROLLOUT_DEVICE = 'cpu'
    TRAIN_DEVICE = 'cpu'
    RB_SIZE = 1_000_000
    LOG_INTERVAL = 100
    START_BUFFER_SIZE = 10_000
    TRAIN_FREQ = 1

    agent = PER_SACAgentMLP(
        state_dim = STATE_DIM,
        action_dim = ACTION_DIM,
        policy_lr = 0.0005,
        q_lr = 0.0005,
        batch_size = 256,
        train_steps = 10,
        gamma = 0.99,
        tau = 0.995,
        alpha = 0.2,
        autotune = True,
        target_entropy_scale = 0.01,
        device = TRAIN_DEVICE,
        log_dir = SAVE_PATH
    )

    # rb = ReplayBuffer(
    #     RB_SIZE,
    #     STATE_DIM,
    #     ACTION_DIM,
    #     TRAIN_DEVICE
    # )

    # rb = PrioritizedReplayBuffer(
    #     alpha=0.5, 
    #     beta=0.4, 
    #     storage=LazyMemmapStorage(RB_SIZE)
    # )

    rb = PrioritizedReplayBuffer(
        RB_SIZE,
        STATE_DIM,
        alpha=0.5,
        beta=0.4,
        beta_frames=int(1e10)
    )

    env = gym.make(ENV_ID)
    start_time = time.time()
    mean_reward, mean_steps = 0, 0
    rollout_actor = copy_model(agent.actor, ROLLOUT_DEVICE)

    for epoch in range(1, int(1e10)):
        state, _ = env.reset()
        terminated, truncated = False, False
        ep_reward, ep_steps = 0, 0
        while not (terminated or truncated):
            action, _, _ = rollout_actor.get_action(torch.tensor(state).to(ROLLOUT_DEVICE))
            next_state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
            # data = {
            #     'state': torch.tensor(state),
            #     'action': action,
            #     'reward': torch.tensor(reward),
            #     'next_state': torch.tensor(next_state),
            #     'done': torch.tensor(terminated or truncated, dtype=torch.float32)
            # }
            rb.add(state, action, reward, next_state, terminated or truncated)
            state = next_state
            ep_reward += reward
            ep_steps += 1

        if agent.log_dir:
            agent.writer.add_scalar("rollout/ep_reward", ep_reward, epoch)
            agent.writer.add_scalar("rollout/ep_steps", ep_steps, epoch)

        mean_reward += ep_reward
        mean_steps += ep_steps

        if epoch % LOG_INTERVAL == 0:
            print(f'epoch: {epoch} mean_reward: {round(mean_reward / LOG_INTERVAL, 2)} mean_steps: {round(mean_steps / LOG_INTERVAL, 2)} buffer_size: {len(rb)} time: {round(time.time() - start_time, 2)} s')
            print(rb.beta)
            mean_reward, mean_steps = 0, 0

        if len(rb) > START_BUFFER_SIZE and epoch % TRAIN_FREQ == 0:
            agent.train(rb)
            rollout_actor = copy_model(agent.actor, ROLLOUT_DEVICE)

    

