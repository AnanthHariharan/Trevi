import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional
import copy
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.skill import Skill, ParametricSkill, SkillLibrary


class MAMLSkillLearner:
    """MAML-based meta-learning for skill acquisition."""
    
    def __init__(self, skill_template: Dict[str, Any], inner_lr: float = 0.01,
                 outer_lr: float = 0.001, inner_steps: int = 5):
        self.skill_template = skill_template
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        
        # Meta-parameters (initialization for new skills)
        self.meta_skill = ParametricSkill(**skill_template)
        self.meta_optimizer = optim.Adam(self.meta_skill.parameters(), lr=outer_lr)
        
        # Training statistics
        self.meta_losses = []
        self.adaptation_losses = []
        
    def meta_train_step(self, task_batch: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Perform one meta-training step on a batch of tasks."""
        meta_gradients = []
        meta_losses = []
        
        for task_data in task_batch:
            # Clone meta-parameters for inner loop
            adapted_skill = self._clone_skill(self.meta_skill)
            
            # Inner loop: adapt to task
            adapted_skill, inner_loss = self._inner_loop_adaptation(adapted_skill, task_data)
            
            # Compute meta-gradient
            meta_loss = self._compute_meta_loss(adapted_skill, task_data)
            meta_losses.append(meta_loss.item())
            
            # Compute gradients w.r.t. meta-parameters
            meta_grads = torch.autograd.grad(
                meta_loss, self.meta_skill.parameters(), 
                create_graph=True, retain_graph=True
            )
            meta_gradients.append(meta_grads)
        
        # Outer loop: update meta-parameters
        self._outer_loop_update(meta_gradients)
        
        avg_meta_loss = np.mean(meta_losses)
        self.meta_losses.append(avg_meta_loss)
        
        return {
            'meta_loss': avg_meta_loss,
            'num_tasks': len(task_batch),
            'inner_steps': self.inner_steps
        }
    
    def _clone_skill(self, skill: ParametricSkill) -> ParametricSkill:
        """Create a deep copy of the skill for adaptation."""
        cloned_skill = ParametricSkill(
            skill_id=skill.skill_id + "_clone",
            name=skill.name,
            input_dim=skill.input_dim,
            output_dim=skill.output_dim
        )
        cloned_skill.load_state_dict(skill.state_dict())
        return cloned_skill
    
    def _inner_loop_adaptation(self, skill: ParametricSkill, 
                              task_data: Dict[str, torch.Tensor]) -> Tuple[ParametricSkill, float]:
        """Adapt skill to specific task using inner loop updates."""
        inner_optimizer = optim.SGD(skill.parameters(), lr=self.inner_lr)
        
        support_states = task_data['support_states']
        support_actions = task_data['support_actions']
        
        total_inner_loss = 0.0
        
        for step in range(self.inner_steps):
            inner_optimizer.zero_grad()
            
            # Forward pass
            predicted_actions = skill(support_states)
            
            # Compute inner loss
            inner_loss = nn.MSELoss()(predicted_actions, support_actions)
            total_inner_loss += inner_loss.item()
            
            # Backward pass and update
            inner_loss.backward()
            inner_optimizer.step()
        
        avg_inner_loss = total_inner_loss / self.inner_steps
        self.adaptation_losses.append(avg_inner_loss)
        
        return skill, avg_inner_loss
    
    def _compute_meta_loss(self, adapted_skill: ParametricSkill, 
                          task_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute meta-loss on query set after adaptation."""
        query_states = task_data['query_states']
        query_actions = task_data['query_actions']
        
        predicted_actions = adapted_skill(query_states)
        meta_loss = nn.MSELoss()(predicted_actions, query_actions)
        
        return meta_loss
    
    def _outer_loop_update(self, meta_gradients: List[Tuple[torch.Tensor, ...]]):
        """Update meta-parameters using averaged gradients."""
        # Average gradients across tasks
        averaged_gradients = []
        for i in range(len(meta_gradients[0])):
            grad_sum = sum(grads[i] for grads in meta_gradients)
            averaged_gradients.append(grad_sum / len(meta_gradients))
        
        # Update meta-parameters
        self.meta_optimizer.zero_grad()
        for param, grad in zip(self.meta_skill.parameters(), averaged_gradients):
            param.grad = grad
        self.meta_optimizer.step()
    
    def adapt_to_new_task(self, task_data: Dict[str, torch.Tensor], 
                         num_adaptation_steps: Optional[int] = None) -> ParametricSkill:
        """Adapt meta-learned skill to a new task."""
        if num_adaptation_steps is None:
            num_adaptation_steps = self.inner_steps
        
        # Create new skill initialized from meta-parameters
        new_skill = self._clone_skill(self.meta_skill)
        new_skill.skill_id = f"adapted_skill_{len(self.adaptation_losses)}"
        
        # Adapt to new task
        adapted_skill, _ = self._inner_loop_adaptation(new_skill, task_data)
        
        return adapted_skill
    
    def generate_skill_for_task(self, task_data: Dict[str, torch.Tensor]) -> ParametricSkill:
        """Generate a new skill for a specific task."""
        return self.adapt_to_new_task(task_data)
    
    def evaluate_meta_learning(self, test_tasks: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Evaluate meta-learning performance on test tasks."""
        adaptation_performances = []
        zero_shot_performances = []
        
        for task_data in test_tasks:
            # Zero-shot performance (no adaptation)
            with torch.no_grad():
                query_states = task_data['query_states']
                query_actions = task_data['query_actions']
                
                zero_shot_predictions = self.meta_skill(query_states)
                zero_shot_loss = nn.MSELoss()(zero_shot_predictions, query_actions).item()
                zero_shot_performances.append(zero_shot_loss)
            
            # Performance after adaptation
            adapted_skill = self.adapt_to_new_task(task_data)
            
            with torch.no_grad():
                adapted_predictions = adapted_skill(query_states)
                adapted_loss = nn.MSELoss()(adapted_predictions, query_actions).item()
                adaptation_performances.append(adapted_loss)
        
        return {
            'zero_shot_loss': np.mean(zero_shot_performances),
            'adapted_loss': np.mean(adaptation_performances),
            'improvement': np.mean(zero_shot_performances) - np.mean(adaptation_performances),
            'num_test_tasks': len(test_tasks)
        }
    
    def save_meta_parameters(self, filepath: str):
        """Save meta-learned parameters."""
        torch.save({
            'meta_skill_state_dict': self.meta_skill.state_dict(),
            'skill_template': self.skill_template,
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'inner_steps': self.inner_steps,
            'meta_losses': self.meta_losses,
            'adaptation_losses': self.adaptation_losses
        }, filepath)
    
    def load_meta_parameters(self, filepath: str):
        """Load meta-learned parameters."""
        checkpoint = torch.load(filepath)
        
        self.meta_skill.load_state_dict(checkpoint['meta_skill_state_dict'])
        self.skill_template = checkpoint['skill_template']
        self.inner_lr = checkpoint['inner_lr']
        self.outer_lr = checkpoint['outer_lr'] 
        self.inner_steps = checkpoint['inner_steps']
        self.meta_losses = checkpoint.get('meta_losses', [])
        self.adaptation_losses = checkpoint.get('adaptation_losses', [])


class SecondOrderMAMLSkillLearner(MAMLSkillLearner):
    """Second-order MAML implementation for skill learning."""
    
    def __init__(self, skill_template: Dict[str, Any], inner_lr: float = 0.01,
                 outer_lr: float = 0.001, inner_steps: int = 5):
        super().__init__(skill_template, inner_lr, outer_lr, inner_steps)
        
    def _inner_loop_adaptation(self, skill: ParametricSkill, 
                              task_data: Dict[str, torch.Tensor]) -> Tuple[ParametricSkill, float]:
        """Second-order adaptation with gradient computation."""
        support_states = task_data['support_states']
        support_actions = task_data['support_actions']
        
        # Store original parameters
        original_params = [p.clone() for p in skill.parameters()]
        
        total_inner_loss = 0.0
        
        for step in range(self.inner_steps):
            # Forward pass
            predicted_actions = skill(support_states)
            
            # Compute inner loss
            inner_loss = nn.MSELoss()(predicted_actions, support_actions)
            total_inner_loss += inner_loss.item()
            
            # Compute gradients
            grads = torch.autograd.grad(inner_loss, skill.parameters(), 
                                      create_graph=True, retain_graph=True)
            
            # Manual parameter update (to maintain computational graph)
            with torch.no_grad():
                for param, grad in zip(skill.parameters(), grads):
                    param.data = param.data - self.inner_lr * grad
        
        avg_inner_loss = total_inner_loss / self.inner_steps
        self.adaptation_losses.append(avg_inner_loss)
        
        return skill, avg_inner_loss


class TaskAgnosticMAMLSkillLearner(MAMLSkillLearner):
    """Task-agnostic MAML for general skill initialization."""
    
    def __init__(self, skill_template: Dict[str, Any], inner_lr: float = 0.01,
                 outer_lr: float = 0.001, inner_steps: int = 5, 
                 diversity_weight: float = 0.1):
        super().__init__(skill_template, inner_lr, outer_lr, inner_steps)
        self.diversity_weight = diversity_weight
        
    def meta_train_step(self, task_batch: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Meta-training with diversity regularization."""
        # Standard MAML update
        standard_metrics = super().meta_train_step(task_batch)
        
        # Add diversity regularization
        diversity_loss = self._compute_diversity_loss(task_batch)
        
        # Update with diversity term
        diversity_grad = torch.autograd.grad(diversity_loss, self.meta_skill.parameters())
        
        with torch.no_grad():
            for param, div_grad in zip(self.meta_skill.parameters(), diversity_grad):
                param.grad += self.diversity_weight * div_grad
        
        self.meta_optimizer.step()
        
        standard_metrics['diversity_loss'] = diversity_loss.item()
        return standard_metrics
    
    def _compute_diversity_loss(self, task_batch: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute diversity regularization term."""
        adapted_skills = []
        
        for task_data in task_batch:
            adapted_skill = self._clone_skill(self.meta_skill)
            adapted_skill, _ = self._inner_loop_adaptation(adapted_skill, task_data)
            adapted_skills.append(adapted_skill)
        
        # Compute pairwise parameter distances
        diversity_loss = torch.tensor(0.0, requires_grad=True)
        
        for i in range(len(adapted_skills)):
            for j in range(i + 1, len(adapted_skills)):
                param_dist = 0.0
                for p1, p2 in zip(adapted_skills[i].parameters(), 
                                 adapted_skills[j].parameters()):
                    param_dist += torch.norm(p1 - p2) ** 2
                
                # Encourage diversity (negative of similarity)
                diversity_loss = diversity_loss - param_dist
        
        return diversity_loss