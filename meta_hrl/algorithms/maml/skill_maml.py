import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
from collections import OrderedDict

from ...core.skill import Skill, ParametricSkill, SkillLibrary


class SkillMAML:
    """Skill-specific MAML implementation with hierarchical adaptation."""
    
    def __init__(self, base_skill_config: Dict[str, Any], 
                 meta_lr: float = 1e-3, adaptation_lr: float = 1e-2,
                 num_adaptation_steps: int = 5, second_order: bool = True):
        self.base_skill_config = base_skill_config
        self.meta_lr = meta_lr
        self.adaptation_lr = adaptation_lr
        self.num_adaptation_steps = num_adaptation_steps
        self.second_order = second_order
        
        # Initialize meta-model
        self.meta_skill = ParametricSkill(**base_skill_config)
        self.meta_optimizer = optim.Adam(self.meta_skill.parameters(), lr=meta_lr)
        
        # Adaptation history for analysis
        self.adaptation_history = []
        self.meta_training_history = []
        
    def fast_adapt(self, skill: ParametricSkill, support_set: Dict[str, torch.Tensor],
                   create_graph: bool = False) -> Tuple[ParametricSkill, List[float]]:
        """Fast adaptation of skill to support set."""
        # Clone skill for adaptation
        adapted_skill = self._create_adapted_skill(skill)
        
        adaptation_losses = []
        
        for step in range(self.num_adaptation_steps):
            # Forward pass on support set
            support_loss = self._compute_support_loss(adapted_skill, support_set)
            adaptation_losses.append(support_loss.item())
            
            # Compute gradients
            gradients = torch.autograd.grad(
                support_loss, 
                adapted_skill.parameters(),
                create_graph=create_graph and self.second_order,
                retain_graph=True
            )
            
            # Update parameters
            self._update_skill_parameters(adapted_skill, gradients)
        
        return adapted_skill, adaptation_losses
    
    def _create_adapted_skill(self, base_skill: ParametricSkill) -> ParametricSkill:
        """Create a copy of skill for adaptation."""
        adapted_skill = ParametricSkill(
            skill_id=base_skill.skill_id + "_adapted",
            name=base_skill.name + "_adapted",
            input_dim=base_skill.input_dim,
            output_dim=base_skill.output_dim
        )
        
        # Copy parameters
        adapted_skill.load_state_dict(base_skill.state_dict())
        
        return adapted_skill
    
    def _compute_support_loss(self, skill: ParametricSkill, 
                             support_set: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss on support set."""
        states = support_set['states']
        actions = support_set['actions']
        
        predicted_actions = skill(states)
        loss = nn.MSELoss()(predicted_actions, actions)
        
        # Add regularization if specified
        if 'regularization_weight' in support_set:
            reg_weight = support_set['regularization_weight']
            reg_loss = sum(torch.norm(p)**2 for p in skill.parameters())
            loss += reg_weight * reg_loss
        
        return loss
    
    def _update_skill_parameters(self, skill: ParametricSkill, gradients: Tuple[torch.Tensor]):
        """Update skill parameters using gradients."""
        with torch.no_grad():
            for param, grad in zip(skill.parameters(), gradients):
                param.data = param.data - self.adaptation_lr * grad
    
    def meta_update(self, task_batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Perform meta-update on batch of tasks."""
        self.meta_optimizer.zero_grad()
        
        total_meta_loss = 0.0
        task_losses = []
        
        for task_data in task_batch:
            support_set = task_data['support']
            query_set = task_data['query']
            
            # Fast adaptation on support set
            adapted_skill, adaptation_losses = self.fast_adapt(
                self.meta_skill, support_set, create_graph=self.second_order
            )
            
            # Compute meta-loss on query set
            query_loss = self._compute_query_loss(adapted_skill, query_set)
            total_meta_loss += query_loss
            task_losses.append(query_loss.item())
        
        # Average meta-loss
        avg_meta_loss = total_meta_loss / len(task_batch)
        
        # Backward pass and meta-update
        avg_meta_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.meta_skill.parameters(), max_norm=1.0)
        
        self.meta_optimizer.step()
        
        # Record training statistics
        meta_stats = {
            'meta_loss': avg_meta_loss.item(),
            'avg_task_loss': np.mean(task_losses),
            'std_task_loss': np.std(task_losses),
            'num_tasks': len(task_batch)
        }
        
        self.meta_training_history.append(meta_stats)
        
        return meta_stats
    
    def _compute_query_loss(self, skill: ParametricSkill, 
                           query_set: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss on query set."""
        states = query_set['states']
        actions = query_set['actions']
        
        predicted_actions = skill(states)
        return nn.MSELoss()(predicted_actions, actions)
    
    def learn_new_skill(self, task_data: Dict[str, Any], 
                       skill_name: str = None) -> ParametricSkill:
        """Learn a new skill from task data."""
        if skill_name is None:
            skill_name = f"learned_skill_{len(self.adaptation_history)}"
        
        # Fast adaptation to create new skill
        support_set = task_data.get('support', task_data)
        adapted_skill, losses = self.fast_adapt(self.meta_skill, support_set)
        
        # Update skill identity
        adapted_skill.skill_id = f"skill_{len(self.adaptation_history)}"
        adapted_skill.name = skill_name
        
        # Record adaptation
        adaptation_record = {
            'skill_name': skill_name,
            'adaptation_losses': losses,
            'final_loss': losses[-1],
            'task_type': task_data.get('task_type', 'unknown')
        }
        self.adaptation_history.append(adaptation_record)
        
        return adapted_skill
    
    def evaluate_adaptation(self, test_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate adaptation performance on test tasks."""
        results = {
            'task_results': [],
            'avg_final_loss': 0.0,
            'avg_improvement': 0.0,
            'success_rate': 0.0
        }
        
        successful_adaptations = 0
        total_final_loss = 0.0
        total_improvement = 0.0
        
        for task_idx, task_data in enumerate(test_tasks):
            support_set = task_data['support']
            query_set = task_data['query']
            
            # Initial performance (no adaptation)
            with torch.no_grad():
                initial_loss = self._compute_query_loss(self.meta_skill, query_set).item()
            
            # Adapted performance
            adapted_skill, adaptation_losses = self.fast_adapt(self.meta_skill, support_set)
            
            with torch.no_grad():
                final_loss = self._compute_query_loss(adapted_skill, query_set).item()
            
            improvement = initial_loss - final_loss
            total_final_loss += final_loss
            total_improvement += improvement
            
            # Consider adaptation successful if improvement > threshold
            if improvement > 0.1 * initial_loss:
                successful_adaptations += 1
            
            task_result = {
                'task_idx': task_idx,
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'improvement': improvement,
                'adaptation_losses': adaptation_losses,
                'successful': improvement > 0.1 * initial_loss
            }
            results['task_results'].append(task_result)
        
        # Aggregate results
        results['avg_final_loss'] = total_final_loss / len(test_tasks)
        results['avg_improvement'] = total_improvement / len(test_tasks)
        results['success_rate'] = successful_adaptations / len(test_tasks)
        
        return results
    
    def get_skill_similarity(self, skill1: ParametricSkill, 
                           skill2: ParametricSkill) -> float:
        """Compute similarity between two skills based on parameters."""
        total_similarity = 0.0
        total_params = 0
        
        for p1, p2 in zip(skill1.parameters(), skill2.parameters()):
            # Cosine similarity between parameter tensors
            p1_flat = p1.view(-1)
            p2_flat = p2.view(-1)
            
            similarity = torch.cosine_similarity(p1_flat, p2_flat, dim=0).item()
            total_similarity += similarity * p1_flat.numel()
            total_params += p1_flat.numel()
        
        return total_similarity / total_params if total_params > 0 else 0.0
    
    def analyze_skill_diversity(self, num_random_tasks: int = 50) -> Dict[str, Any]:
        """Analyze diversity of adapted skills."""
        # Generate random tasks for diversity analysis
        random_tasks = self._generate_random_tasks(num_random_tasks)
        
        adapted_skills = []
        for task_data in random_tasks:
            adapted_skill, _ = self.fast_adapt(self.meta_skill, task_data['support'])
            adapted_skills.append(adapted_skill)
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(adapted_skills)):
            for j in range(i + 1, len(adapted_skills)):
                sim = self.get_skill_similarity(adapted_skills[i], adapted_skills[j])
                similarities.append(sim)
        
        return {
            'num_skills': len(adapted_skills),
            'avg_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'diversity_score': 1.0 - np.mean(similarities)  # Higher is more diverse
        }
    
    def _generate_random_tasks(self, num_tasks: int) -> List[Dict[str, Any]]:
        """Generate random tasks for testing."""
        tasks = []
        
        for _ in range(num_tasks):
            # Generate random support and query sets
            batch_size = np.random.randint(5, 20)
            input_dim = self.base_skill_config['input_dim']
            output_dim = self.base_skill_config['output_dim']
            
            support_states = torch.randn(batch_size, input_dim)
            support_actions = torch.randn(batch_size, output_dim)
            
            query_states = torch.randn(batch_size, input_dim)
            query_actions = torch.randn(batch_size, output_dim)
            
            task = {
                'support': {
                    'states': support_states,
                    'actions': support_actions
                },
                'query': {
                    'states': query_states,
                    'actions': query_actions
                }
            }
            tasks.append(task)
        
        return tasks
    
    def save_checkpoint(self, filepath: str):
        """Save MAML checkpoint."""
        checkpoint = {
            'meta_skill_state_dict': self.meta_skill.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'base_skill_config': self.base_skill_config,
            'hyperparameters': {
                'meta_lr': self.meta_lr,
                'adaptation_lr': self.adaptation_lr,
                'num_adaptation_steps': self.num_adaptation_steps,
                'second_order': self.second_order
            },
            'training_history': self.meta_training_history,
            'adaptation_history': self.adaptation_history
        }
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load MAML checkpoint."""
        checkpoint = torch.load(filepath)
        
        self.meta_skill.load_state_dict(checkpoint['meta_skill_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        
        # Restore hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.meta_lr = hyperparams['meta_lr']
        self.adaptation_lr = hyperparams['adaptation_lr']
        self.num_adaptation_steps = hyperparams['num_adaptation_steps']
        self.second_order = hyperparams['second_order']
        
        # Restore training history
        self.meta_training_history = checkpoint.get('training_history', [])
        self.adaptation_history = checkpoint.get('adaptation_history', [])


class ContextualSkillMAML(SkillMAML):
    """MAML extension with contextual embeddings for skill adaptation."""
    
    def __init__(self, base_skill_config: Dict[str, Any], context_dim: int = 32,
                 **kwargs):
        super().__init__(base_skill_config, **kwargs)
        self.context_dim = context_dim
        
        # Context encoder
        input_dim = base_skill_config['input_dim']
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 64),  # state + action
            nn.ReLU(),
            nn.Linear(64, context_dim),
            nn.Tanh()
        )
        
        # Context-conditioned skill adaptation
        self.context_adapter = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, sum(p.numel() for p in self.meta_skill.parameters()))
        )
        
        # Update optimizer to include new parameters
        all_params = (list(self.meta_skill.parameters()) + 
                     list(self.context_encoder.parameters()) +
                     list(self.context_adapter.parameters()))
        self.meta_optimizer = optim.Adam(all_params, lr=self.meta_lr)
    
    def encode_context(self, support_set: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode task context from support set."""
        states = support_set['states']
        actions = support_set['actions']
        
        # Combine states and actions for context
        state_action_pairs = torch.cat([states, actions], dim=1)
        
        # Average over support examples to get task context
        context_embeddings = self.context_encoder(state_action_pairs)
        task_context = context_embeddings.mean(dim=0)
        
        return task_context
    
    def fast_adapt_with_context(self, skill: ParametricSkill, 
                               support_set: Dict[str, torch.Tensor],
                               create_graph: bool = False) -> Tuple[ParametricSkill, torch.Tensor, List[float]]:
        """Fast adaptation with contextual information."""
        # Encode task context
        task_context = self.encode_context(support_set)
        
        # Generate context-based parameter adjustments
        context_adjustments = self.context_adapter(task_context)
        
        # Apply adjustments to skill parameters
        adapted_skill = self._create_context_adapted_skill(skill, context_adjustments)
        
        # Standard fast adaptation
        adapted_skill, adaptation_losses = self.fast_adapt(adapted_skill, support_set, create_graph)
        
        return adapted_skill, task_context, adaptation_losses
    
    def _create_context_adapted_skill(self, base_skill: ParametricSkill, 
                                     adjustments: torch.Tensor) -> ParametricSkill:
        """Create skill with context-based parameter adjustments."""
        adapted_skill = self._create_adapted_skill(base_skill)
        
        # Apply context adjustments
        param_idx = 0
        with torch.no_grad():
            for param in adapted_skill.parameters():
                param_size = param.numel()
                adjustment = adjustments[param_idx:param_idx + param_size]
                param.data += 0.1 * adjustment.view(param.shape)  # Small adjustment
                param_idx += param_size
        
        return adapted_skill