import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from abc import ABC, abstractmethod

from .skill import Skill, SkillLibrary, ParametricSkill


class MetaLearner(ABC):
    """Abstract base class for meta-learning algorithms."""
    
    def __init__(self, skill_library: SkillLibrary, learning_rate: float = 1e-3):
        self.skill_library = skill_library
        self.learning_rate = learning_rate
        self.meta_step = 0
        
    @abstractmethod
    def meta_update(self, tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Perform meta-update given a batch of tasks."""
        pass
    
    @abstractmethod
    def adapt_skill(self, skill: Skill, task_data: Dict[str, Any], 
                   adaptation_steps: int = 5) -> Skill:
        """Adapt a skill to a new task."""
        pass
    
    @abstractmethod
    def generate_new_skill(self, task_data: Dict[str, Any]) -> Skill:
        """Generate a new skill for a given task."""
        pass


class MAMLSkillLearner(MetaLearner):
    """MAML-based meta-learner for skills."""
    
    def __init__(self, skill_library: SkillLibrary, learning_rate: float = 1e-3,
                 meta_learning_rate: float = 1e-4, adaptation_steps: int = 5):
        super().__init__(skill_library, learning_rate)
        self.meta_learning_rate = meta_learning_rate
        self.adaptation_steps = adaptation_steps
        self.skill_template = None
        
    def set_skill_template(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        """Set the neural network template for new skills."""
        self.skill_template = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_dims': hidden_dims
        }
    
    def meta_update(self, tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """MAML meta-update across multiple tasks."""
        if not self.skill_template:
            raise ValueError("Skill template not set. Call set_skill_template first.")
        
        meta_losses = []
        meta_gradients = []
        
        for task in tasks:
            # Create a temporary skill for this task
            temp_skill = ParametricSkill(
                skill_id=f"temp_{self.meta_step}",
                name="temporary",
                **self.skill_template
            )
            
            # Fast adaptation
            adapted_skill = self._fast_adapt(temp_skill, task)
            
            # Compute meta-loss
            meta_loss = self._compute_meta_loss(adapted_skill, task)
            meta_losses.append(meta_loss.item())
            
            # Compute gradients w.r.t. original parameters
            meta_grads = torch.autograd.grad(meta_loss, temp_skill.parameters(), 
                                           create_graph=True)
            meta_gradients.append(meta_grads)
        
        # Average gradients and update meta-parameters
        avg_meta_loss = np.mean(meta_losses)
        self._update_meta_parameters(meta_gradients)
        
        self.meta_step += 1
        return {'meta_loss': avg_meta_loss, 'num_tasks': len(tasks)}
    
    def _fast_adapt(self, skill: ParametricSkill, task_data: Dict[str, Any]) -> ParametricSkill:
        """Perform fast adaptation on a skill."""
        adapted_skill = ParametricSkill(
            skill_id=skill.skill_id + "_adapted",
            name=skill.name,
            input_dim=skill.input_dim,
            output_dim=skill.output_dim
        )
        
        # Copy parameters
        adapted_skill.load_state_dict(skill.state_dict())
        
        optimizer = optim.SGD(adapted_skill.parameters(), lr=self.learning_rate)
        
        states = task_data['states']
        actions = task_data['actions']
        
        for _ in range(self.adaptation_steps):
            optimizer.zero_grad()
            predictions = adapted_skill(states)
            loss = nn.MSELoss()(predictions, actions)
            loss.backward()
            optimizer.step()
        
        return adapted_skill
    
    def _compute_meta_loss(self, skill: ParametricSkill, task_data: Dict[str, Any]) -> torch.Tensor:
        """Compute meta-loss for adapted skill."""
        meta_states = task_data.get('meta_states', task_data['states'])
        meta_actions = task_data.get('meta_actions', task_data['actions'])
        
        predictions = skill(meta_states)
        return nn.MSELoss()(predictions, meta_actions)
    
    def _update_meta_parameters(self, meta_gradients: List[Tuple[torch.Tensor, ...]]):
        """Update meta-parameters using averaged gradients."""
        # In practice, this would update a meta-network or skill generator
        pass
    
    def adapt_skill(self, skill: Skill, task_data: Dict[str, Any], 
                   adaptation_steps: int = 5) -> Skill:
        """Adapt an existing skill to new task."""
        if not isinstance(skill, ParametricSkill):
            raise ValueError("Can only adapt parametric skills")
        
        return self._fast_adapt(skill, task_data)
    
    def generate_new_skill(self, task_data: Dict[str, Any]) -> Skill:
        """Generate a new skill based on meta-learned initialization."""
        if not self.skill_template:
            raise ValueError("Skill template not set")
        
        new_skill = ParametricSkill(
            skill_id=f"skill_{self.meta_step}_{len(self.skill_library)}",
            name=f"learned_skill_{len(self.skill_library)}",
            **self.skill_template
        )
        
        # Initialize with meta-learned parameters (would be learned in practice)
        # For now, use random initialization
        
        return self.adapt_skill(new_skill, task_data, self.adaptation_steps)


class HierarchicalMetaLearner(MetaLearner):
    """Meta-learner that operates at multiple hierarchical levels."""
    
    def __init__(self, skill_library: SkillLibrary, learning_rate: float = 1e-3,
                 hierarchy_levels: int = 3):
        super().__init__(skill_library, learning_rate)
        self.hierarchy_levels = hierarchy_levels
        self.level_learners = {}
        
        # Initialize meta-learners for each level
        for level in range(hierarchy_levels):
            self.level_learners[level] = MAMLSkillLearner(skill_library, learning_rate)
    
    def meta_update(self, tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Hierarchical meta-update."""
        total_losses = {}
        
        for level in range(self.hierarchy_levels):
            # Filter tasks appropriate for this hierarchy level
            level_tasks = self._filter_tasks_by_level(tasks, level)
            
            if level_tasks:
                level_metrics = self.level_learners[level].meta_update(level_tasks)
                total_losses[f'level_{level}_loss'] = level_metrics['meta_loss']
        
        return total_losses
    
    def _filter_tasks_by_level(self, tasks: List[Dict[str, Any]], level: int) -> List[Dict[str, Any]]:
        """Filter tasks appropriate for hierarchy level."""
        # Simple heuristic: longer horizon tasks for higher levels
        min_horizon = level * 10
        max_horizon = (level + 1) * 10
        
        filtered = []
        for task in tasks:
            horizon = task.get('horizon', 0)
            if min_horizon <= horizon < max_horizon:
                filtered.append(task)
        
        return filtered
    
    def adapt_skill(self, skill: Skill, task_data: Dict[str, Any], 
                   adaptation_steps: int = 5) -> Skill:
        """Adapt skill using appropriate hierarchy level."""
        horizon = task_data.get('horizon', 0)
        level = min(horizon // 10, self.hierarchy_levels - 1)
        
        return self.level_learners[level].adapt_skill(skill, task_data, adaptation_steps)
    
    def generate_new_skill(self, task_data: Dict[str, Any]) -> Skill:
        """Generate new skill at appropriate hierarchy level."""
        horizon = task_data.get('horizon', 0)
        level = min(horizon // 10, self.hierarchy_levels - 1)
        
        return self.level_learners[level].generate_new_skill(task_data)