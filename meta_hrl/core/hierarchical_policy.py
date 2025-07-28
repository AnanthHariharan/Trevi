import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

from .skill import Skill, SkillLibrary
from .skill_composer import SkillComposer


class PolicyLevel(Enum):
    """Hierarchical policy levels."""
    HIGH = "high"      # Long-term planning, goal setting
    MID = "mid"        # Skill selection and sequencing  
    LOW = "low"        # Primitive action execution


class HierarchicalPolicy:
    """Multi-level hierarchical policy for skill-based control."""
    
    def __init__(self, skill_library: SkillLibrary, skill_composer: SkillComposer,
                 state_dim: int, action_dim: int, goal_dim: int):
        self.skill_library = skill_library
        self.skill_composer = skill_composer
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        
        # Initialize policy networks for each level
        self.high_level_policy = HighLevelPolicy(state_dim, goal_dim)
        self.mid_level_policy = MidLevelPolicy(state_dim, goal_dim, skill_library)
        self.low_level_policy = LowLevelPolicy(state_dim, action_dim)
        
        # Current execution state
        self.current_goal = None
        self.current_skill_sequence = []
        self.current_skill_index = 0
        self.current_skill = None
        self.skill_start_time = 0
        
    def act(self, state: torch.Tensor, timestep: int = 0) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Execute hierarchical policy to get action."""
        info = {}
        
        # High-level: Set goals (executed less frequently)
        if self._should_update_goal(timestep):
            self.current_goal = self.high_level_policy.select_goal(state)
            info['new_goal'] = True
            info['goal'] = self.current_goal
        
        # Mid-level: Select skills and compose sequences
        if self._should_update_skill_sequence(state, timestep):
            self.current_skill_sequence = self.skill_composer.compose_skills(
                self.current_goal, state
            )
            self.current_skill_index = 0
            info['new_skill_sequence'] = True
            info['skill_sequence'] = self.current_skill_sequence
        
        # Select current skill from sequence
        if self._should_update_current_skill(state, timestep):
            if self.current_skill_index < len(self.current_skill_sequence):
                skill_id = self.current_skill_sequence[self.current_skill_index]
                self.current_skill = self.skill_library.get_skill(skill_id)
                self.skill_start_time = timestep
                info['new_skill'] = True
                info['current_skill'] = skill_id
            else:
                # Sequence completed, need new goal
                self.current_goal = None
                self.current_skill = None
        
        # Low-level: Execute current skill or primitive action
        if self.current_skill is not None:
            action = self.current_skill.forward(state)
            info['skill_action'] = True
            info['skill_id'] = self.current_skill.skill_id
        else:
            # Fallback to low-level policy
            action = self.low_level_policy.forward(state)
            info['primitive_action'] = True
        
        return action, info
    
    def _should_update_goal(self, timestep: int, goal_update_frequency: int = 50) -> bool:
        """Check if high-level goal should be updated."""
        return (self.current_goal is None or 
                timestep % goal_update_frequency == 0)
    
    def _should_update_skill_sequence(self, state: torch.Tensor, timestep: int) -> bool:
        """Check if skill sequence should be updated."""
        if self.current_goal is None:
            return False
        
        # Update if no current sequence or goal changed significantly
        if not self.current_skill_sequence:
            return True
        
        # Check if current sequence is still valid
        if self.current_skill_index >= len(self.current_skill_sequence):
            return True
        
        return False
    
    def _should_update_current_skill(self, state: torch.Tensor, timestep: int) -> bool:
        """Check if current skill should be updated."""
        if not self.current_skill_sequence:
            return False
        
        # Update if no current skill
        if self.current_skill is None:
            return True
        
        # Check termination condition
        if self.current_skill.termination_condition(state):
            self.current_skill_index += 1
            return True
        
        # Check timeout (prevent infinite skill execution)
        skill_duration = timestep - self.skill_start_time
        if skill_duration > 100:  # Max skill duration
            self.current_skill_index += 1
            return True
        
        return False
    
    def update_skill_success(self, success: bool):
        """Update success rate of currently executed skill."""
        if self.current_skill is not None:
            self.current_skill.update_success_rate(success)
    
    def get_current_hierarchy_state(self) -> Dict[str, Any]:
        """Get current state of hierarchical execution."""
        return {
            'current_goal': self.current_goal,
            'skill_sequence': self.current_skill_sequence,
            'skill_index': self.current_skill_index,
            'current_skill': self.current_skill.skill_id if self.current_skill else None,
            'sequence_progress': (self.current_skill_index / max(1, len(self.current_skill_sequence)))
        }


class HighLevelPolicy(nn.Module):
    """High-level policy for goal selection."""
    
    def __init__(self, state_dim: int, goal_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        
        self.goal_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),
            nn.Tanh()  # Normalize goal to [-1, 1]
        )
        
        # Goal value network for goal selection
        self.value_network = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def select_goal(self, state: torch.Tensor, num_candidates: int = 5) -> torch.Tensor:
        """Select goal using value-based selection."""
        with torch.no_grad():
            # Generate multiple goal candidates
            goal_candidates = []
            values = []
            
            for _ in range(num_candidates):
                goal = self.goal_network(state + 0.1 * torch.randn_like(state))
                value = self.value_network(torch.cat([state, goal]))
                
                goal_candidates.append(goal)
                values.append(value.item())
            
            # Select goal with highest value
            best_idx = np.argmax(values)
            return goal_candidates[best_idx]
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Generate goal from current state."""
        return self.goal_network(state)


class MidLevelPolicy(nn.Module):
    """Mid-level policy for skill selection."""
    
    def __init__(self, state_dim: int, goal_dim: int, skill_library: SkillLibrary,
                 hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.skill_library = skill_library
        
        # Skill selection network
        self.skill_selector = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # Skill embedding dimension
        )
        
        # Skill embedding network
        self.skill_embedder = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),  # Input/output dimensions
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def select_skill(self, state: torch.Tensor, goal: torch.Tensor) -> Optional[str]:
        """Select best skill for current state and goal."""
        available_skills = self.skill_library.get_available_skills(state)
        
        if not available_skills:
            return None
        
        with torch.no_grad():
            # Get selection embedding
            selection_embedding = self.skill_selector(torch.cat([state, goal]))
            
            # Compute similarity scores with available skills
            best_skill = None
            best_score = float('-inf')
            
            for skill in available_skills:
                skill_embedding = self._get_skill_embedding(skill)
                similarity = torch.cosine_similarity(
                    selection_embedding, skill_embedding, dim=0
                ).item()
                
                # Combine with skill success rate
                score = 0.7 * similarity + 0.3 * skill.success_rate
                
                if score > best_score:
                    best_score = score
                    best_skill = skill
            
            return best_skill.skill_id if best_skill else None
    
    def _get_skill_embedding(self, skill: Skill) -> torch.Tensor:
        """Get embedding for a skill."""
        # Use skill input/output dimensions as simple embedding
        skill_repr = torch.tensor([
            float(skill.input_dim), float(skill.output_dim)
        ] + [0.0] * (self.state_dim * 2 - 2))
        
        return self.skill_embedder(skill_repr)
    
    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Generate skill selection probabilities."""
        return self.skill_selector(torch.cat([state, goal]))


class LowLevelPolicy(nn.Module):
    """Low-level policy for primitive action execution."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.action_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Normalize actions
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Generate primitive action from state."""
        return self.action_network(state)


class HierarchicalPolicyTrainer:
    """Trainer for hierarchical policy components."""
    
    def __init__(self, hierarchical_policy: HierarchicalPolicy, learning_rates: Dict[str, float]):
        self.policy = hierarchical_policy
        
        # Initialize optimizers for each level
        self.optimizers = {
            'high': torch.optim.Adam(
                self.policy.high_level_policy.parameters(),
                lr=learning_rates.get('high', 1e-4)
            ),
            'mid': torch.optim.Adam(
                self.policy.mid_level_policy.parameters(),
                lr=learning_rates.get('mid', 1e-3)
            ),
            'low': torch.optim.Adam(
                self.policy.low_level_policy.parameters(),
                lr=learning_rates.get('low', 1e-3)
            )
        }
    
    def train_step(self, batch_data: Dict[str, torch.Tensor], level: str) -> Dict[str, float]:
        """Perform training step for specific policy level."""
        if level not in self.optimizers:
            raise ValueError(f"Unknown policy level: {level}")
        
        optimizer = self.optimizers[level]
        
        if level == 'high':
            return self._train_high_level(batch_data, optimizer)
        elif level == 'mid':
            return self._train_mid_level(batch_data, optimizer)
        elif level == 'low':
            return self._train_low_level(batch_data, optimizer)
    
    def _train_high_level(self, batch_data: Dict[str, torch.Tensor], 
                         optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Train high-level goal selection policy."""
        states = batch_data['states']
        target_goals = batch_data['target_goals']
        goal_values = batch_data.get('goal_values', torch.zeros(len(states)))
        
        optimizer.zero_grad()
        
        # Goal generation loss
        predicted_goals = self.policy.high_level_policy(states)
        goal_loss = nn.MSELoss()(predicted_goals, target_goals)
        
        # Value prediction loss
        state_goal_pairs = torch.cat([states, predicted_goals], dim=1)
        predicted_values = self.policy.high_level_policy.value_network(state_goal_pairs)
        value_loss = nn.MSELoss()(predicted_values.squeeze(), goal_values)
        
        total_loss = goal_loss + value_loss
        total_loss.backward()
        optimizer.step()
        
        return {
            'goal_loss': goal_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _train_mid_level(self, batch_data: Dict[str, torch.Tensor],
                        optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Train mid-level skill selection policy."""
        states = batch_data['states']
        goals = batch_data['goals']
        target_skills = batch_data['target_skill_embeddings']
        
        optimizer.zero_grad()
        
        predicted_embeddings = self.policy.mid_level_policy(states, goals)
        loss = nn.MSELoss()(predicted_embeddings, target_skills)
        
        loss.backward()
        optimizer.step()
        
        return {'skill_selection_loss': loss.item()}
    
    def _train_low_level(self, batch_data: Dict[str, torch.Tensor],
                        optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Train low-level primitive action policy."""
        states = batch_data['states']
        target_actions = batch_data['actions']
        
        optimizer.zero_grad()
        
        predicted_actions = self.policy.low_level_policy(states)
        loss = nn.MSELoss()(predicted_actions, target_actions)
        
        loss.backward()
        optimizer.step()
        
        return {'action_loss': loss.item()}