#!/usr/bin/env python3
"""
Demo script showing how to use the Meta-HRL framework with visualization.
"""

import sys
import os

# Add the parent directory to Python path so we can import from the main project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import asyncio
import json
import time

# Import Meta-HRL components
from core.skill import SkillLibrary, ParametricSkill
from core.meta_learner import MAMLSkillLearner
from core.skill_composer import SkillComposer
from core.hierarchical_policy import HierarchicalPolicy
from algorithms.maml.skill_maml import SkillMAML
from utils.visualization import SkillVisualization, TrainingVisualization

# For dashboard integration
import requests
import threading


class MetaHRLDemo:
    """Demonstration of Meta-HRL framework with visualization."""
    
    def __init__(self):
        self.skill_library = SkillLibrary()
        self.skill_visualizer = SkillVisualization(self.skill_library)
        self.training_visualizer = TrainingVisualization()
        
        # Initialize meta-learner
        self.meta_learner = SkillMAML({
            'skill_id': 'meta_skill',
            'name': 'Meta Skill Template',
            'input_dim': 10,
            'output_dim': 4,
            'hidden_dims': [64, 64]
        })
        
        # Initialize skill composer
        self.skill_composer = SkillComposer(self.skill_library)
        
        # Training metrics for dashboard
        self.training_metrics = {}
        self.dashboard_url = "http://localhost:8000/api"
        
    def create_sample_skills(self):
        """Create some sample skills for demonstration."""
        print("Creating sample skills...")
        
        # Create different types of skills
        skill_configs = [
            {"name": "Move Forward", "input_dim": 10, "output_dim": 4},
            {"name": "Turn Left", "input_dim": 10, "output_dim": 4},
            {"name": "Turn Right", "input_dim": 10, "output_dim": 4},
            {"name": "Pick Object", "input_dim": 10, "output_dim": 4},
            {"name": "Place Object", "input_dim": 10, "output_dim": 4},
        ]
        
        for i, config in enumerate(skill_configs):
            skill = ParametricSkill(
                skill_id=f"skill_{i}",
                name=config["name"],
                input_dim=config["input_dim"],
                output_dim=config["output_dim"]
            )
            
            # Simulate training by setting success rates
            skill.success_rate = np.random.beta(3, 2)  # Biased towards higher success
            skill.is_trained = True
            
            # Create random embedding
            embedding = torch.randn(32)
            
            self.skill_library.add_skill(skill, embedding)
        
        # Create some composite skills
        self.skill_library.set_composition("navigate_and_pick", ["skill_0", "skill_1", "skill_3"])
        self.skill_library.set_composition("pick_and_place", ["skill_3", "skill_4"])
        
        print(f"Created {len(self.skill_library)} skills")
    
    def simulate_training(self, num_iterations=100):
        """Simulate meta-learning training with live updates."""
        print(f"Starting simulated training for {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            # Simulate training metrics
            meta_loss = 2.0 * np.exp(-iteration / 30) + 0.1 * np.random.randn()
            adaptation_loss = 1.5 * np.exp(-iteration / 25) + 0.1 * np.random.randn()
            success_rate = 1 - np.exp(-iteration / 40) + 0.05 * np.random.randn()
            
            # Store metrics
            metrics = {
                'meta_loss': max(0, meta_loss),
                'adaptation_loss': max(0, adaptation_loss),
                'success_rate': max(0, min(1, success_rate)),
                'iteration': iteration,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update local metrics
            for key, value in metrics.items():
                if key not in self.training_metrics:
                    self.training_metrics[key] = []
                self.training_metrics[key].append(value)
            
            # Send to dashboard if available
            try:
                response = requests.post(
                    f"{self.dashboard_url}/training/update-metrics",
                    json=metrics,
                    timeout=1
                )
                if iteration % 10 == 0:
                    print(f"Iteration {iteration}: Loss={meta_loss:.3f}, Success={success_rate:.3f}")
            except requests.exceptions.RequestException:
                # Dashboard not available, continue without it
                if iteration % 10 == 0:
                    print(f"Iteration {iteration}: Loss={meta_loss:.3f}, Success={success_rate:.3f} (Dashboard offline)")
            
            # Update skill success rates
            for skill in self.skill_library:
                if np.random.rand() < 0.1:  # 10% chance to update
                    skill.success_rate = min(1.0, skill.success_rate + 0.01 * np.random.randn())
            
            time.sleep(0.1)  # Simulate training time
    
    def demonstrate_skill_composition(self):
        """Demonstrate skill composition capabilities."""
        print("\nDemonstrating skill composition...")
        
        # Create a mock state and goal
        current_state = torch.randn(10)
        goal_state = torch.randn(10)
        
        # Compose skills to reach goal
        skill_sequence = self.skill_composer.compose_skills(goal_state, current_state)
        
        print(f"Generated skill sequence: {skill_sequence}")
        
        # Explain the composition
        if skill_sequence:
            explanation = self.skill_composer.explain_composition(skill_sequence)
            print(f"Composition explanation:")
            print(f"  - Sequence length: {explanation['length']}")
            print(f"  - Estimated success rate: {explanation['estimated_success_rate']:.3f}")
            
            for component in explanation['components']:
                print(f"  - {component['name']}: {component['success_rate']:.3f} success rate")
    
    def create_visualizations(self):
        """Create and save visualization plots."""
        print("\nCreating visualizations...")
        
        # Skill library overview
        fig = self.skill_visualizer.plot_skill_library_overview()
        fig.savefig('skill_library_overview.png', dpi=300, bbox_inches='tight')
        print("Saved: skill_library_overview.png")
        
        # Training dashboard
        if self.training_metrics:
            fig = self.training_visualizer.plot_training_dashboard(self.training_metrics)
            fig.savefig('training_dashboard.png', dpi=300, bbox_inches='tight')
            print("Saved: training_dashboard.png")
        
        # Interactive skill network (save as HTML)
        try:
            interactive_fig = self.skill_visualizer.create_interactive_skill_network()
            interactive_fig.write_html('interactive_skill_network.html')
            print("Saved: interactive_skill_network.html")
        except Exception as e:
            print(f"Could not create interactive plot: {e}")
        
        plt.close('all')  # Clean up matplotlib figures
    
    def setup_hierarchical_policy(self):
        """Setup and demonstrate hierarchical policy."""
        print("\nSetting up hierarchical policy...")
        
        # Create hierarchical policy
        self.hierarchical_policy = HierarchicalPolicy(
            skill_library=self.skill_library,
            skill_composer=self.skill_composer,
            state_dim=10,
            action_dim=4,
            goal_dim=10
        )
        
        # Simulate some execution steps
        current_state = torch.randn(10)
        
        for step in range(10):
            action, info = self.hierarchical_policy.act(current_state, timestep=step)
            
            if step % 3 == 0:  # Print every few steps
                hierarchy_state = self.hierarchical_policy.get_current_hierarchy_state()
                print(f"Step {step}: Current skill = {hierarchy_state.get('current_skill', 'None')}")
            
            # Simulate state transition
            current_state = current_state + 0.1 * torch.randn(10)
    
    def run_dashboard_integration(self):
        """Test dashboard integration."""
        print("\nTesting dashboard integration...")
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.dashboard_url}/health", timeout=2)
            if response.status_code == 200:
                print("✓ Dashboard is running and accessible")
                
                # Send some sample skills to dashboard
                for skill in self.skill_library.skills.values():
                    skill_data = {
                        "id": skill.skill_id,
                        "name": skill.name,
                        "input_dim": skill.input_dim,
                        "output_dim": skill.output_dim
                    }
                    
                    try:
                        requests.post(f"{self.dashboard_url}/skills", json=skill_data, timeout=1)
                    except:
                        pass  # Skill might already exist
                
                print("✓ Skills synchronized with dashboard")
                
            else:
                print("✗ Dashboard is not responding correctly")
                
        except requests.exceptions.RequestException:
            print("✗ Dashboard is not running. Start it with:")
            print("  cd frontend/backend && python main.py")
    
    def run_complete_demo(self):
        """Run the complete demonstration."""
        print("=" * 60)
        print("Meta-Learning Hierarchical Skill Acquisition Demo")
        print("=" * 60)
        
        # Step 1: Create sample skills
        self.create_sample_skills()
        
        # Step 2: Test dashboard integration
        self.run_dashboard_integration()
        
        # Step 3: Setup hierarchical policy
        self.setup_hierarchical_policy()
        
        # Step 4: Demonstrate skill composition
        self.demonstrate_skill_composition()
        
        # Step 5: Run simulated training
        print("\n" + "="*40)
        print("Starting Training Simulation")
        print("="*40)
        self.simulate_training(num_iterations=50)
        
        # Step 6: Create visualizations
        self.create_visualizations()
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60)
        print("\nGenerated files:")
        print("  - skill_library_overview.png")
        print("  - training_dashboard.png") 
        print("  - interactive_skill_network.html")
        print("\nTo view the interactive dashboard:")
        print("  1. cd frontend/backend && python main.py")
        print("  2. cd frontend/frontend && npm start")
        print("  3. Open http://localhost:3000")


def main():
    """Main function to run the demo."""
    demo = MetaHRLDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()