import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class Skill(ABC):
    """Abstract base class for primitive skills."""
    
    def __init__(self, skill_id: str, name: str, input_dim: int, output_dim: int):
        self.skill_id = skill_id
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_trained = False
        self.success_rate = 0.0
        
    @abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Execute the skill given current state."""
        pass
    
    @abstractmethod
    def can_execute(self, state: torch.Tensor) -> bool:
        """Check if skill can be executed in current state."""
        pass
    
    @abstractmethod
    def termination_condition(self, state: torch.Tensor) -> bool:
        """Check if skill should terminate."""
        pass
    
    def update_success_rate(self, success: bool):
        """Update skill success rate using exponential moving average."""
        alpha = 0.1
        self.success_rate = alpha * float(success) + (1 - alpha) * self.success_rate


class ParametricSkill(Skill, nn.Module):
    """Neural network-based parametric skill."""
    
    def __init__(self, skill_id: str, name: str, input_dim: int, output_dim: int, 
                 hidden_dims: List[int] = [64, 64]):
        Skill.__init__(self, skill_id, name, input_dim, output_dim)
        nn.Module.__init__(self)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        self.termination_network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
    
    def can_execute(self, state: torch.Tensor) -> bool:
        return True  # Override in subclasses for specific preconditions
    
    def termination_condition(self, state: torch.Tensor) -> bool:
        with torch.no_grad():
            return self.termination_network(state).item() > 0.5


class SkillLibrary:
    """Library to store and manage learned skills."""
    
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.skill_embeddings: Dict[str, torch.Tensor] = {}
        self.composition_graph: Dict[str, List[str]] = {}
        
    def add_skill(self, skill: Skill, embedding: Optional[torch.Tensor] = None):
        """Add a new skill to the library."""
        self.skills[skill.skill_id] = skill
        if embedding is not None:
            self.skill_embeddings[skill.skill_id] = embedding
        self.composition_graph[skill.skill_id] = []
        
    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Retrieve a skill by ID."""
        return self.skills.get(skill_id)
    
    def get_available_skills(self, state: torch.Tensor) -> List[Skill]:
        """Get all skills that can be executed in the current state."""
        available = []
        for skill in self.skills.values():
            if skill.can_execute(state):
                available.append(skill)
        return available
    
    def get_composition_chain(self, skill_id: str) -> List[str]:
        """Get the composition chain for a composite skill."""
        return self.composition_graph.get(skill_id, [])
    
    def set_composition(self, composite_skill_id: str, component_skills: List[str]):
        """Define how a composite skill is composed of other skills."""
        self.composition_graph[composite_skill_id] = component_skills
    
    def get_skill_similarity(self, skill1_id: str, skill2_id: str) -> float:
        """Compute similarity between two skills based on embeddings."""
        if skill1_id not in self.skill_embeddings or skill2_id not in self.skill_embeddings:
            return 0.0
        
        emb1 = self.skill_embeddings[skill1_id]
        emb2 = self.skill_embeddings[skill2_id]
        
        return torch.cosine_similarity(emb1, emb2, dim=0).item()
    
    def prune_skills(self, min_success_rate: float = 0.1):
        """Remove skills with low success rates."""
        to_remove = []
        for skill_id, skill in self.skills.items():
            if skill.success_rate < min_success_rate:
                to_remove.append(skill_id)
        
        for skill_id in to_remove:
            del self.skills[skill_id]
            if skill_id in self.skill_embeddings:
                del self.skill_embeddings[skill_id]
            if skill_id in self.composition_graph:
                del self.composition_graph[skill_id]
    
    def __len__(self) -> int:
        return len(self.skills)
    
    def __iter__(self):
        return iter(self.skills.values())