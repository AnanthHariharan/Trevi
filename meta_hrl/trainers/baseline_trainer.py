# meta_hrl/trainers/baseline_trainer.py
import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from ..utils.networks import PolicyNetwork
from ..environments.simple_navigation import SimpleNavigation

print("Baseline Trainer loaded")