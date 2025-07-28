import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from ...core.skill import Skill, SkillLibrary


class OptionCriticAgent:
    """Option-Critic algorithm for hierarchical skill learning."""
    
    def __init__(self, state_dim: int, action_dim: int, num_options: int,
                 learning_rate: float = 1e-3, gamma: float = 0.99,
                 termination_reg: float = 0.01, entropy_reg: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options
        self.gamma = gamma
        self.termination_reg = termination_reg
        self.entropy_reg = entropy_reg
        
        # Networks
        self.option_critic_network = OptionCriticNetwork(
            state_dim, action_dim, num_options
        )
        
        # Optimizers
        self.optimizer = optim.Adam(self.option_critic_network.parameters(), lr=learning_rate)
        
        # Current option
        self.current_option = None
        self.option_history = []
        
        # Training statistics
        self.training_stats = {
            'critic_losses': [],
            'actor_losses': [],
            'termination_losses': [],
            'option_lengths': []
        }
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, int]:
        """Select action using option-critic policy."""
        with torch.no_grad():
            # Option selection
            if self.current_option is None or self._should_terminate_option(state):
                self.current_option = self._select_option(state)
            
            # Action selection within option
            action_probs = self.option_critic_network.get_action_probs(state, self.current_option)
            
            if deterministic:
                action = action_probs.argmax()
            else:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
            
            return action, self.current_option
    
    def _select_option(self, state: torch.Tensor) -> int:
        """Select option using option value function."""
        with torch.no_grad():
            option_values = self.option_critic_network.get_option_values(state)
            return option_values.argmax().item()
    
    def _should_terminate_option(self, state: torch.Tensor) -> bool:
        """Check if current option should terminate."""
        if self.current_option is None:
            return True
        
        with torch.no_grad():
            termination_prob = self.option_critic_network.get_termination_prob(
                state, self.current_option
            )
            return torch.bernoulli(termination_prob).item() > 0.5
    
    def update(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update option-critic networks."""
        states = batch_data['states']
        actions = batch_data['actions']
        rewards = batch_data['rewards']
        next_states = batch_data['next_states']
        dones = batch_data['dones']
        options = batch_data['options']
        
        # Compute losses
        critic_loss = self._compute_critic_loss(states, actions, rewards, next_states, dones, options)
        actor_loss = self._compute_actor_loss(states, actions, options)
        termination_loss = self._compute_termination_loss(states, next_states, options)
        
        # Total loss
        total_loss = critic_loss + actor_loss + termination_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.option_critic_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Record statistics
        self.training_stats['critic_losses'].append(critic_loss.item())
        self.training_stats['actor_losses'].append(actor_loss.item())
        self.training_stats['termination_losses'].append(termination_loss.item())
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'termination_loss': termination_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _compute_critic_loss(self, states: torch.Tensor, actions: torch.Tensor,
                           rewards: torch.Tensor, next_states: torch.Tensor,
                           dones: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        """Compute option-value function loss."""
        # Current option values
        current_option_values = self.option_critic_network.get_option_values_batch(states, options)
        
        # Target computation
        with torch.no_grad():
            next_option_values = self.option_critic_network.get_option_values(next_states)
            next_termination_probs = self.option_critic_network.get_termination_probs(next_states, options)
            
            # Option-value target with termination
            next_values = ((1 - next_termination_probs) * next_option_values.gather(1, options.unsqueeze(1)).squeeze() +
                          next_termination_probs * next_option_values.max(dim=1)[0])
            
            targets = rewards + self.gamma * (1 - dones) * next_values
        
        critic_loss = F.mse_loss(current_option_values, targets)
        return critic_loss
    
    def _compute_actor_loss(self, states: torch.Tensor, actions: torch.Tensor,
                          options: torch.Tensor) -> torch.Tensor:
        """Compute policy loss for each option."""
        # Get action log probabilities
        action_log_probs = self.option_critic_network.get_action_log_probs_batch(states, actions, options)
        
        # Compute advantages
        with torch.no_grad():
            option_values = self.option_critic_network.get_option_values_batch(states, options)
            state_values = self.option_critic_network.get_state_values(states)
            advantages = option_values - state_values
        
        # Policy loss with entropy regularization
        policy_loss = -(action_log_probs * advantages).mean()
        
        # Entropy regularization
        action_probs = self.option_critic_network.get_action_probs_batch(states, options)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
        
        actor_loss = policy_loss - self.entropy_reg * entropy
        return actor_loss
    
    def _compute_termination_loss(self, states: torch.Tensor, next_states: torch.Tensor,
                                options: torch.Tensor) -> torch.Tensor:
        """Compute termination function loss."""
        # Current termination probabilities
        termination_probs = self.option_critic_network.get_termination_probs(states, options)
        
        # Compute termination advantages
        with torch.no_grad():
            current_option_values = self.option_critic_network.get_option_values_batch(states, options)
            next_state_values = self.option_critic_network.get_state_values(next_states)
            termination_advantages = next_state_values - current_option_values
        
        # Termination loss with regularization
        termination_loss = (termination_probs * termination_advantages).mean()
        
        # Add regularization to encourage termination
        termination_reg_loss = self.termination_reg * termination_probs.mean()
        
        return termination_loss + termination_reg_loss
    
    def get_option_statistics(self) -> Dict[str, Any]:
        """Get statistics about option usage."""
        if not self.option_history:
            return {}
        
        option_counts = np.bincount(self.option_history, minlength=self.num_options)
        option_frequencies = option_counts / len(self.option_history)
        
        return {
            'option_frequencies': option_frequencies.tolist(),
            'most_used_option': int(np.argmax(option_counts)),
            'least_used_option': int(np.argmin(option_counts)),
            'option_entropy': -np.sum(option_frequencies * np.log(option_frequencies + 1e-8)),
            'total_options_used': len(self.option_history)
        }
    
    def extract_skills_from_options(self) -> List[Skill]:
        """Extract learned skills from option policies."""
        skills = []
        
        for option_idx in range(self.num_options):
            # Create skill from option policy
            skill = OptionSkill(
                skill_id=f"option_skill_{option_idx}",
                name=f"Option {option_idx} Skill",
                input_dim=self.state_dim,
                output_dim=self.action_dim,
                option_idx=option_idx,
                option_critic_network=self.option_critic_network
            )
            skills.append(skill)
        
        return skills


class OptionCriticNetwork(nn.Module):
    """Neural networks for Option-Critic algorithm."""
    
    def __init__(self, state_dim: int, action_dim: int, num_options: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Option-value function Q(s, ω)
        self.option_value_head = nn.Linear(hidden_dim, num_options)
        
        # Intra-option policies π(a|s, ω)
        self.option_policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
            ) for _ in range(num_options)
        ])
        
        # Termination functions β(s, ω)
        self.termination_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_options)
        ])
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through all networks."""
        features = self.feature_extractor(state)
        
        # Option values
        option_values = self.option_value_head(features)
        
        # Option policies
        option_policies = torch.stack([
            policy(features) for policy in self.option_policies
        ], dim=1)
        
        # Termination probabilities
        termination_probs = torch.stack([
            termination(features).squeeze(-1) for termination in self.termination_networks
        ], dim=1)
        
        return {
            'option_values': option_values,
            'option_policies': option_policies,
            'termination_probs': termination_probs
        }
    
    def get_option_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get option values for state."""
        features = self.feature_extractor(state)
        return self.option_value_head(features)
    
    def get_option_values_batch(self, states: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        """Get option values for specific options in batch."""
        option_values = self.get_option_values(states)
        return option_values.gather(1, options.unsqueeze(1)).squeeze()
    
    def get_state_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get state values (max over options)."""
        option_values = self.get_option_values(state)
        return option_values.max(dim=-1)[0]
    
    def get_action_probs(self, state: torch.Tensor, option: int) -> torch.Tensor:
        """Get action probabilities for specific option."""
        features = self.feature_extractor(state)
        return self.option_policies[option](features)
    
    def get_action_probs_batch(self, states: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        """Get action probabilities for batch of states and options."""
        features = self.feature_extractor(states)
        batch_size = states.size(0)
        
        action_probs = []
        for i in range(batch_size):
            option_idx = options[i].item()
            probs = self.option_policies[option_idx](features[i])
            action_probs.append(probs)
        
        return torch.stack(action_probs)
    
    def get_action_log_probs_batch(self, states: torch.Tensor, actions: torch.Tensor,
                                  options: torch.Tensor) -> torch.Tensor:
        """Get action log probabilities for batch."""
        action_probs = self.get_action_probs_batch(states, options)
        selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        return torch.log(selected_probs + 1e-8)
    
    def get_termination_prob(self, state: torch.Tensor, option: int) -> torch.Tensor:
        """Get termination probability for specific option."""
        features = self.feature_extractor(state)
        return self.termination_networks[option](features).squeeze()
    
    def get_termination_probs(self, states: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        """Get termination probabilities for batch."""
        features = self.feature_extractor(states)
        batch_size = states.size(0)
        
        termination_probs = []
        for i in range(batch_size):
            option_idx = options[i].item()
            prob = self.termination_networks[option_idx](features[i]).squeeze()
            termination_probs.append(prob)
        
        return torch.stack(termination_probs)


class OptionSkill(Skill):
    """Skill extracted from option-critic option."""
    
    def __init__(self, skill_id: str, name: str, input_dim: int, output_dim: int,
                 option_idx: int, option_critic_network: OptionCriticNetwork):
        super().__init__(skill_id, name, input_dim, output_dim)
        self.option_idx = option_idx
        self.option_critic_network = option_critic_network
        self.is_trained = True
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Execute option policy."""
        with torch.no_grad():
            action_probs = self.option_critic_network.get_action_probs(state, self.option_idx)
            # Return action with highest probability (deterministic)
            return action_probs.argmax().float()
    
    def can_execute(self, state: torch.Tensor) -> bool:
        """Always executable (could add preconditions)."""
        return True
    
    def termination_condition(self, state: torch.Tensor) -> bool:
        """Use learned termination function."""
        with torch.no_grad():
            termination_prob = self.option_critic_network.get_termination_prob(
                state, self.option_idx
            )
            return termination_prob.item() > 0.5