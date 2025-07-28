import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import deque

from .skill import Skill, SkillLibrary


class SkillComposer:
    """Manages composition and sequencing of primitive skills."""
    
    def __init__(self, skill_library: SkillLibrary, max_composition_depth: int = 5):
        self.skill_library = skill_library
        self.max_composition_depth = max_composition_depth
        self.composition_history = deque(maxlen=1000)
        self.successful_compositions = {}
        
    def compose_skills(self, goal_state: torch.Tensor, current_state: torch.Tensor,
                      max_steps: int = 20) -> List[str]:
        """Compose a sequence of skills to reach goal from current state."""
        skill_sequence = []
        state = current_state.clone()
        
        for step in range(max_steps):
            # Get available skills for current state
            available_skills = self.skill_library.get_available_skills(state)
            
            if not available_skills:
                break
                
            # Select best skill using composition policy
            selected_skill = self._select_skill(available_skills, state, goal_state)
            
            if selected_skill is None:
                break
                
            skill_sequence.append(selected_skill.skill_id)
            
            # Simulate skill execution (in practice, would execute in environment)
            state = self._simulate_skill_execution(selected_skill, state)
            
            # Check if goal reached
            if self._goal_reached(state, goal_state):
                break
                
            # Check termination condition
            if selected_skill.termination_condition(state):
                continue
                
        return skill_sequence
    
    def _select_skill(self, available_skills: List[Skill], current_state: torch.Tensor,
                     goal_state: torch.Tensor) -> Optional[Skill]:
        """Select the best skill for current situation."""
        if not available_skills:
            return None
        
        # Simple heuristic: select skill that moves closest to goal
        best_skill = None
        best_score = float('-inf')
        
        for skill in available_skills:
            score = self._compute_skill_score(skill, current_state, goal_state)
            if score > best_score:
                best_score = score
                best_skill = skill
        
        return best_skill
    
    def _compute_skill_score(self, skill: Skill, current_state: torch.Tensor,
                           goal_state: torch.Tensor) -> float:
        """Compute score for skill selection."""
        # Combine multiple factors
        success_score = skill.success_rate
        
        # Distance-based score (simulate skill execution)
        simulated_state = self._simulate_skill_execution(skill, current_state)
        distance_score = -torch.norm(simulated_state - goal_state).item()
        
        # Composition history score
        history_score = self._get_composition_history_score(skill.skill_id)
        
        # Weighted combination
        total_score = (0.4 * success_score + 
                      0.4 * distance_score + 
                      0.2 * history_score)
        
        return total_score
    
    def _simulate_skill_execution(self, skill: Skill, state: torch.Tensor) -> torch.Tensor:
        """Simulate execution of skill (placeholder)."""
        # In practice, this would use learned dynamics models
        with torch.no_grad():
            action = skill.forward(state)
            # Simple state transition model
            next_state = state + 0.1 * action + 0.01 * torch.randn_like(state)
            return next_state
    
    def _goal_reached(self, current_state: torch.Tensor, goal_state: torch.Tensor,
                     threshold: float = 0.1) -> bool:
        """Check if goal state is reached."""
        return torch.norm(current_state - goal_state).item() < threshold
    
    def _get_composition_history_score(self, skill_id: str) -> float:
        """Get score based on composition history."""
        if skill_id in self.successful_compositions:
            return self.successful_compositions[skill_id] / max(1, len(self.composition_history))
        return 0.0
    
    def record_composition_result(self, skill_sequence: List[str], success: bool):
        """Record the result of a skill composition."""
        self.composition_history.append({
            'sequence': skill_sequence,
            'success': success
        })
        
        if success:
            for skill_id in skill_sequence:
                if skill_id not in self.successful_compositions:
                    self.successful_compositions[skill_id] = 0
                self.successful_compositions[skill_id] += 1
    
    def discover_new_compositions(self, successful_sequences: List[List[str]],
                                min_frequency: int = 3) -> Dict[str, List[str]]:
        """Discover frequently occurring skill patterns."""
        pattern_counts = {}
        new_compositions = {}
        
        for sequence in successful_sequences:
            for length in range(2, min(len(sequence) + 1, self.max_composition_depth + 1)):
                for start in range(len(sequence) - length + 1):
                    pattern = tuple(sequence[start:start + length])
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Create new composite skills for frequent patterns
        for pattern, count in pattern_counts.items():
            if count >= min_frequency and len(pattern) > 1:
                composite_id = f"composite_{'_'.join(pattern)}"
                new_compositions[composite_id] = list(pattern)
                
                # Add to skill library as composition
                self.skill_library.set_composition(composite_id, list(pattern))
        
        return new_compositions
    
    def optimize_composition(self, skill_sequence: List[str]) -> List[str]:
        """Optimize a skill sequence by removing redundant skills."""
        if len(skill_sequence) <= 1:
            return skill_sequence
        
        optimized = []
        i = 0
        
        while i < len(skill_sequence):
            current_skill_id = skill_sequence[i]
            
            # Look for composite skills that match upcoming sequence
            best_match_length = 1
            best_composite_id = None
            
            for composite_id, component_skills in self.skill_library.composition_graph.items():
                if (i + len(component_skills) <= len(skill_sequence) and
                    component_skills == skill_sequence[i:i + len(component_skills)] and
                    len(component_skills) > best_match_length):
                    best_match_length = len(component_skills)
                    best_composite_id = composite_id
            
            if best_composite_id:
                optimized.append(best_composite_id)
                i += best_match_length
            else:
                optimized.append(current_skill_id)
                i += 1
        
        return optimized
    
    def explain_composition(self, skill_sequence: List[str]) -> Dict[str, Any]:
        """Provide explanation for skill composition."""
        explanation = {
            'sequence': skill_sequence,
            'length': len(skill_sequence),
            'components': [],
            'estimated_success_rate': 1.0
        }
        
        for skill_id in skill_sequence:
            skill = self.skill_library.get_skill(skill_id)
            if skill:
                component_info = {
                    'skill_id': skill_id,
                    'name': skill.name,
                    'success_rate': skill.success_rate,
                    'is_composite': skill_id in self.skill_library.composition_graph
                }
                
                if component_info['is_composite']:
                    component_info['sub_skills'] = self.skill_library.get_composition_chain(skill_id)
                
                explanation['components'].append(component_info)
                explanation['estimated_success_rate'] *= skill.success_rate
        
        return explanation


class AttentionBasedComposer(SkillComposer):
    """Skill composer using attention mechanism for skill selection."""
    
    def __init__(self, skill_library: SkillLibrary, state_dim: int, 
                 attention_dim: int = 64, max_composition_depth: int = 5):
        super().__init__(skill_library, max_composition_depth)
        self.state_dim = state_dim
        self.attention_dim = attention_dim
        
        # Attention network for skill selection
        self.attention_net = nn.Sequential(
            nn.Linear(state_dim * 2, attention_dim),  # current + goal state
            nn.ReLU(),
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1)
        )
        
        self.skill_encoder = nn.Sequential(
            nn.Linear(state_dim * 2, attention_dim),  # skill representation
            nn.ReLU(),
            nn.Linear(attention_dim, attention_dim)
        )
    
    def _select_skill(self, available_skills: List[Skill], current_state: torch.Tensor,
                     goal_state: torch.Tensor) -> Optional[Skill]:
        """Select skill using attention mechanism."""
        if not available_skills:
            return None
        
        state_goal = torch.cat([current_state, goal_state])
        
        attention_scores = []
        for skill in available_skills:
            # Create skill representation (placeholder)
            skill_repr = self._get_skill_representation(skill, current_state)
            
            # Compute attention score
            attention_input = torch.cat([state_goal, skill_repr])
            score = self.attention_net(attention_input).item()
            attention_scores.append(score)
        
        # Select skill with highest attention score
        best_idx = np.argmax(attention_scores)
        return available_skills[best_idx]
    
    def _get_skill_representation(self, skill: Skill, state: torch.Tensor) -> torch.Tensor:
        """Get representation of skill for attention computation."""
        # Placeholder: use skill output as representation
        with torch.no_grad():
            skill_output = skill.forward(state)
            # Pad or truncate to match expected dimension
            if skill_output.size(0) < self.state_dim:
                padding = torch.zeros(self.state_dim - skill_output.size(0))
                skill_repr = torch.cat([skill_output, padding])
            else:
                skill_repr = skill_output[:self.state_dim]
            
            # Duplicate to match state_goal dimension
            return torch.cat([skill_repr, skill_repr])
    
    def train_attention(self, training_data: List[Dict[str, Any]], epochs: int = 100):
        """Train the attention mechanism."""
        optimizer = torch.optim.Adam(
            list(self.attention_net.parameters()) + list(self.skill_encoder.parameters()),
            lr=1e-3
        )
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for data in training_data:
                current_state = data['current_state']
                goal_state = data['goal_state']
                optimal_skill_id = data['optimal_skill']
                available_skills = data['available_skills']
                
                # Find optimal skill in available skills
                optimal_skill = None
                for skill in available_skills:
                    if skill.skill_id == optimal_skill_id:
                        optimal_skill = skill
                        break
                
                if optimal_skill is None:
                    continue
                
                # Compute attention scores
                state_goal = torch.cat([current_state, goal_state])
                
                scores = []
                for skill in available_skills:
                    skill_repr = self._get_skill_representation(skill, current_state)
                    attention_input = torch.cat([state_goal, skill_repr])
                    score = self.attention_net(attention_input)
                    scores.append(score)
                
                scores = torch.stack(scores)
                
                # Create target (one-hot for optimal skill)
                target = torch.zeros_like(scores)
                optimal_idx = available_skills.index(optimal_skill)
                target[optimal_idx] = 1.0
                
                # Compute loss
                loss = nn.CrossEntropyLoss()(scores.unsqueeze(0), target.argmax().unsqueeze(0))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"Attention training epoch {epoch}, loss: {total_loss / len(training_data):.4f}")