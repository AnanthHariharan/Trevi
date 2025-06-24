# meta_hrl/environments/base_env.py
import abc
import gymnasium as gym
import torch

class BaseEnv(gym.Env, abc.ABC):
    """
    Abstract base class for all environments in this project.
    It extends the standard Gymnasium API to include task-related methods.
    """
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def sample_tasks(self, n_tasks):
        """Samples a batch of tasks from the environment's task distribution."""
        pass

    @abc.abstractmethod
    def set_task(self, task):
        """Sets the environment to a specific task."""
        pass