import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class OptionTerminationNetwork(nn.Module):
    """Learnable termination conditions for options."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute termination probability."""
        return self.network(state).squeeze(-1)
    
    def should_terminate(self, state: torch.Tensor, threshold: float = 0.5) -> bool:
        """Binary termination decision."""
        with torch.no_grad():
            prob = self.forward(state)
            return prob.item() > threshold