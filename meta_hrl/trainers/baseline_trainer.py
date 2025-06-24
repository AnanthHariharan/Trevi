# meta_hrl/trainers/baseline_trainer.py
import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from ..utils.networks import PolicyNetwork
from ..environments.simple_navigation import SimpleNavigation

def train_baseline():
    """A simple REINFORCE training loop for a single task."""
    print("--- Training Baseline Agent ---")
    
    # Config (in a real scenario, this would be loaded from a YAML file)
    env_config = {"grid_size": 10, "max_steps": 50}
    lr = 1e-3
    hidden_dim = 64
    num_episodes = 500
    gamma = 0.99

    # Setup
    env = SimpleNavigation(env_config)
    task = env.sample_tasks(1)[0] # Get one fixed task
    env.set_task(task)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = PolicyNetwork(obs_dim, hidden_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    for i_episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []

        # Collect trajectory
        while True:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action_probs = policy(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            state = next_state
            
            if terminated or truncated:
                break
        
        # REINFORCE update
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        optimizer.step()

        if i_episode % 50 == 0:
            print(f"Episode {i_episode}, Last Reward: {rewards[-1]:.2f}, Total Reward: {sum(rewards):.2f}")

    print("--- Baseline Training Complete ---")