# meta_hrl/environments/simple_navigation.py
import numpy as np
from gymnasium import spaces
from .base_env import BaseEnv

class SimpleNavigation(BaseEnv):
    """
    A simple 2D grid navigation environment.
    The "task" is defined by the goal position.
    """
    def __init__(self, config):
        super().__init__()
        self.grid_size = config.get("grid_size", 10)
        self.max_steps = config.get("max_steps", 50)
        self.current_step = 0
        self.agent_pos = None
        self.goal_pos = None

        # Observation space: agent_x, agent_y, goal_x, goal_y
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(4,), dtype=np.float32)
        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)

    def sample_tasks(self, n_tasks):
        return [self.observation_space.sample()[2:] for _ in range(n_tasks)]

    def set_task(self, task):
        self.goal_pos = np.array(task, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # For simplicity, start agent at a fixed position
        self.agent_pos = np.array([self.grid_size / 2, self.grid_size / 2], dtype=np.float32)
        
        # If no goal is set, sample one
        if self.goal_pos is None:
            self.goal_pos = self.observation_space.sample()[2:]

        obs = np.concatenate([self.agent_pos, self.goal_pos])
        info = {}
        return obs, info

    def step(self, action):
        if action == 0:  # Up
            self.agent_pos[1] += 1
        elif action == 1: # Down
            self.agent_pos[1] -= 1
        elif action == 2: # Left
            self.agent_pos[0] -= 1
        elif action == 3: # Right
            self.agent_pos[0] += 1

        # Clip to be within grid boundaries
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size - 1)
        self.current_step += 1

        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward = -dist_to_goal  # Reward is negative distance to goal
        
        terminated = dist_to_goal < 1.0  # Episode ends if agent is close to the goal
        truncated = self.current_step >= self.max_steps

        obs = np.concatenate([self.agent_pos, self.goal_pos])
        info = {}

        return obs, reward, terminated, truncated, info