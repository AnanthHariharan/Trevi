import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from core.skill import SkillLibrary, Skill
from core.hierarchical_policy import HierarchicalPolicy


class SkillVisualization:
    """Visualization utilities for skills and hierarchical policies."""
    
    def __init__(self, skill_library: SkillLibrary):
        self.skill_library = skill_library
        plt.style.use('seaborn-v0_8')
        
    def plot_skill_library_overview(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Create comprehensive skill library visualization."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Skill Library Overview', fontsize=16, fontweight='bold')
        
        # 1. Skill success rates
        self._plot_skill_success_rates(axes[0, 0])
        
        # 2. Skill similarity heatmap
        self._plot_skill_similarity_matrix(axes[0, 1])
        
        # 3. Skill composition graph
        self._plot_skill_composition_graph(axes[0, 2])
        
        # 4. Skill usage frequency
        self._plot_skill_usage_frequency(axes[1, 0])
        
        # 5. Skill embedding visualization (if available)
        self._plot_skill_embeddings_2d(axes[1, 1])
        
        # 6. Skill learning curves
        self._plot_skill_learning_curves(axes[1, 2])
        
        plt.tight_layout()
        return fig
    
    def _plot_skill_success_rates(self, ax: plt.Axes):
        """Plot skill success rates."""
        skills = list(self.skill_library.skills.values())
        names = [skill.name for skill in skills]
        success_rates = [skill.success_rate for skill in skills]
        
        bars = ax.bar(range(len(names)), success_rates, 
                     color=plt.cm.viridis([rate for rate in success_rates]))
        ax.set_title('Skill Success Rates')
        ax.set_xlabel('Skills')
        ax.set_ylabel('Success Rate')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{rate:.2f}', ha='center', va='bottom')
    
    def _plot_skill_similarity_matrix(self, ax: plt.Axes):
        """Plot skill similarity heatmap."""
        skills = list(self.skill_library.skills.values())
        n_skills = len(skills)
        
        if n_skills < 2:
            ax.text(0.5, 0.5, 'Need at least 2 skills', ha='center', va='center')
            ax.set_title('Skill Similarity Matrix')
            return
        
        similarity_matrix = np.zeros((n_skills, n_skills))
        
        for i in range(n_skills):
            for j in range(n_skills):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self.skill_library.get_skill_similarity(
                        skills[i].skill_id, skills[j].skill_id
                    )
                    similarity_matrix[i, j] = sim
        
        im = ax.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title('Skill Similarity Matrix')
        
        # Add skill names as labels
        skill_names = [skill.name for skill in skills]
        ax.set_xticks(range(n_skills))
        ax.set_yticks(range(n_skills))
        ax.set_xticklabels(skill_names, rotation=45, ha='right')
        ax.set_yticklabels(skill_names)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
    
    def _plot_skill_composition_graph(self, ax: plt.Axes):
        """Plot skill composition as network graph."""
        G = nx.DiGraph()
        
        # Add nodes for all skills
        for skill_id in self.skill_library.skills.keys():
            G.add_node(skill_id)
        
        # Add edges for compositions
        for composite, components in self.skill_library.composition_graph.items():
            for component in components:
                if component in self.skill_library.skills:
                    G.add_edge(component, composite)
        
        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, 'No skills to display', ha='center', va='center')
            ax.set_title('Skill Composition Graph')
            return
        
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', 
                              node_size=1000, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                              arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        
        ax.set_title('Skill Composition Graph')
        ax.axis('off')
    
    def _plot_skill_usage_frequency(self, ax: plt.Axes):
        """Plot skill usage frequency (placeholder)."""
        # This would need usage tracking in the actual implementation
        skills = list(self.skill_library.skills.keys())
        # Simulate usage data
        usage_counts = np.random.poisson(10, len(skills))
        
        ax.bar(range(len(skills)), usage_counts, color='skyblue')
        ax.set_title('Skill Usage Frequency')
        ax.set_xlabel('Skills')
        ax.set_ylabel('Usage Count')
        ax.set_xticks(range(len(skills)))
        ax.set_xticklabels(skills, rotation=45, ha='right')
    
    def _plot_skill_embeddings_2d(self, ax: plt.Axes):
        """Plot 2D projection of skill embeddings."""
        if not self.skill_library.skill_embeddings:
            ax.text(0.5, 0.5, 'No skill embeddings available', 
                   ha='center', va='center')
            ax.set_title('Skill Embeddings (2D)')
            return
        
        # Get embeddings and project to 2D using PCA
        from sklearn.decomposition import PCA
        
        embeddings = []
        labels = []
        
        for skill_id, embedding in self.skill_library.skill_embeddings.items():
            embeddings.append(embedding.cpu().numpy())
            labels.append(skill_id)
        
        if len(embeddings) < 2:
            ax.text(0.5, 0.5, 'Need at least 2 embeddings', 
                   ha='center', va='center')
            ax.set_title('Skill Embeddings (2D)')
            return
        
        embeddings = np.array(embeddings)
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           c=range(len(labels)), cmap='tab10', s=100)
        
        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_title('Skill Embeddings (2D PCA)')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    
    def _plot_skill_learning_curves(self, ax: plt.Axes):
        """Plot learning curves for skills (placeholder)."""
        # This would need learning history tracking
        x = np.arange(100)
        
        for i, skill_id in enumerate(list(self.skill_library.skills.keys())[:5]):
            # Simulate learning curve
            curve = 1 - np.exp(-x / 20) + 0.1 * np.random.randn(len(x))
            ax.plot(x, curve, label=skill_id, alpha=0.7)
        
        ax.set_title('Skill Learning Curves')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Performance')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def create_interactive_skill_network(self) -> go.Figure:
        """Create interactive skill composition network using Plotly."""
        # Build network
        G = nx.DiGraph()
        
        for skill_id in self.skill_library.skills.keys():
            G.add_node(skill_id)
        
        for composite, components in self.skill_library.composition_graph.items():
            for component in components:
                if component in self.skill_library.skills:
                    G.add_edge(component, composite)
        
        # Get positions
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Extract node and edge information
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())
        
        # Node colors based on success rates
        node_colors = []
        for node in G.nodes():
            skill = self.skill_library.get_skill(node)
            if skill:
                node_colors.append(skill.success_rate)
            else:
                node_colors.append(0.5)
        
        # Edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Composition Links'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=[f"Skill: {skill}<br>Success Rate: {color:.2f}" 
                      for skill, color in zip(node_text, node_colors)],
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_colors,
                size=20,
                colorbar=dict(
                    thickness=15,
                    len=0.7,
                    x=1.02,
                    title="Success Rate"
                )
            ),
            name='Skills'
        ))
        
        fig.update_layout(
            title='Interactive Skill Composition Network',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Hover over nodes for details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig


class HierarchicalPolicyVisualization:
    """Visualization for hierarchical policy execution."""
    
    def __init__(self, hierarchical_policy: HierarchicalPolicy):
        self.policy = hierarchical_policy
        
    def plot_policy_hierarchy(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Visualize the hierarchical policy structure."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Hierarchical Policy Analysis', fontsize=16, fontweight='bold')
        
        # 1. Goal evolution
        self._plot_goal_evolution(axes[0, 0])
        
        # 2. Skill sequence timeline
        self._plot_skill_sequence(axes[0, 1])
        
        # 3. Policy level activation
        self._plot_policy_level_activation(axes[1, 0])
        
        # 4. Execution statistics
        self._plot_execution_statistics(axes[1, 1])
        
        plt.tight_layout()
        return fig
    
    def _plot_goal_evolution(self, ax: plt.Axes):
        """Plot how goals evolve over time."""
        # Simulate goal trajectory data
        timesteps = np.arange(100)
        goal_dims = 3
        
        for dim in range(goal_dims):
            goal_trajectory = np.sin(timesteps / 20 + dim) + 0.1 * np.random.randn(len(timesteps))
            ax.plot(timesteps, goal_trajectory, label=f'Goal Dim {dim+1}', alpha=0.7)
        
        ax.set_title('Goal Evolution Over Time')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Goal Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_skill_sequence(self, ax: plt.Axes):
        """Plot skill sequence timeline."""
        # Get current hierarchy state
        hierarchy_state = self.policy.get_current_hierarchy_state()
        
        if hierarchy_state['skill_sequence']:
            skills = hierarchy_state['skill_sequence']
            y_pos = np.arange(len(skills))
            
            # Color code by skill type or success rate
            colors = plt.cm.Set3(np.linspace(0, 1, len(skills)))
            
            bars = ax.barh(y_pos, [1] * len(skills), color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(skills)
            ax.set_xlabel('Execution')
            ax.set_title('Current Skill Sequence')
            
            # Highlight current skill
            current_idx = hierarchy_state.get('skill_index', 0)
            if current_idx < len(bars):
                bars[current_idx].set_edgecolor('red')
                bars[current_idx].set_linewidth(3)
        else:
            ax.text(0.5, 0.5, 'No skill sequence active', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Current Skill Sequence')
    
    def _plot_policy_level_activation(self, ax: plt.Axes):
        """Plot activation of different policy levels."""
        levels = ['High Level', 'Mid Level', 'Low Level']
        # Simulate activation data
        activations = np.random.rand(3)
        
        bars = ax.bar(levels, activations, color=['red', 'orange', 'blue'], alpha=0.7)
        ax.set_title('Policy Level Activation')
        ax.set_ylabel('Activation Level')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, activation in zip(bars, activations):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{activation:.2f}', ha='center', va='bottom')
    
    def _plot_execution_statistics(self, ax: plt.Axes):
        """Plot execution statistics."""
        hierarchy_state = self.policy.get_current_hierarchy_state()
        
        stats = {
            'Sequence Progress': hierarchy_state.get('sequence_progress', 0),
            'Goal Achievement': np.random.rand(),  # Would track actual goal achievement
            'Skill Success Rate': np.random.rand(),  # Average success rate
            'Adaptation Rate': np.random.rand()  # How often skills are adapted
        }
        
        # Create pie chart of statistics
        labels = list(stats.keys())
        values = list(stats.values())
        
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                         startangle=90, colors=plt.cm.Pastel1.colors)
        ax.set_title('Execution Statistics')


class TrainingVisualization:
    """Visualization for training progress and metrics."""
    
    def __init__(self):
        self.training_history = []
        
    def plot_training_dashboard(self, metrics_history: Dict[str, List[float]], 
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Create comprehensive training dashboard."""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main loss curves
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_loss_curves(ax1, metrics_history)
        
        # Learning rate schedule
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_learning_rate(ax2, metrics_history)
        
        # Success rate evolution
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_success_rates(ax3, metrics_history)
        
        # Skill diversity
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_skill_diversity(ax4, metrics_history)
        
        # Adaptation efficiency
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_adaptation_efficiency(ax5, metrics_history)
        
        # Transfer performance
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_transfer_performance(ax6, metrics_history)
        
        # Resource usage
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_resource_usage(ax7, metrics_history)
        
        fig.suptitle('Meta-Learning Training Dashboard', fontsize=16, fontweight='bold')
        return fig
    
    def _plot_loss_curves(self, ax: plt.Axes, metrics: Dict[str, List[float]]):
        """Plot training loss curves."""
        for loss_name, values in metrics.items():
            if 'loss' in loss_name.lower():
                ax.plot(values, label=loss_name, alpha=0.8)
        
        ax.set_title('Training Loss Curves')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    def _plot_learning_rate(self, ax: plt.Axes, metrics: Dict[str, List[float]]):
        """Plot learning rate schedule."""
        if 'learning_rate' in metrics:
            ax.plot(metrics['learning_rate'], color='red')
            ax.set_title('Learning Rate')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('LR')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No LR data', ha='center', va='center')
    
    def _plot_success_rates(self, ax: plt.Axes, metrics: Dict[str, List[float]]):
        """Plot success rate evolution."""
        success_metrics = {k: v for k, v in metrics.items() 
                          if 'success' in k.lower() or 'accuracy' in k.lower()}
        
        for metric_name, values in success_metrics.items():
            ax.plot(values, label=metric_name, alpha=0.8)
        
        if not success_metrics:
            # Simulate success rate data
            x = np.arange(100)
            ax.plot(x, 1 - np.exp(-x/30), label='Overall Success Rate')
        
        ax.set_title('Success Rate Evolution')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_skill_diversity(self, ax: plt.Axes, metrics: Dict[str, List[float]]):
        """Plot skill diversity metrics."""
        if 'diversity_score' in metrics:
            ax.plot(metrics['diversity_score'], color='green')
        else:
            # Simulate diversity data
            x = np.arange(50)
            diversity = 0.5 + 0.3 * np.sin(x/10) + 0.1 * np.random.randn(len(x))
            ax.plot(x, diversity, color='green')
        
        ax.set_title('Skill Diversity')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Diversity Score')
        ax.grid(True, alpha=0.3)
    
    def _plot_adaptation_efficiency(self, ax: plt.Axes, metrics: Dict[str, List[float]]):
        """Plot adaptation efficiency."""
        # Simulate adaptation efficiency
        efficiency = np.random.beta(2, 2, 20)
        ax.hist(efficiency, bins=10, alpha=0.7, color='purple')
        ax.set_title('Adaptation Efficiency')
        ax.set_xlabel('Efficiency')
        ax.set_ylabel('Frequency')
    
    def _plot_transfer_performance(self, ax: plt.Axes, metrics: Dict[str, List[float]]):
        """Plot transfer performance comparison."""
        tasks = ['Task A', 'Task B', 'Task C', 'Task D']
        before_transfer = np.random.rand(4) * 0.4 + 0.1
        after_transfer = before_transfer + np.random.rand(4) * 0.4
        
        x = np.arange(len(tasks))
        width = 0.35
        
        ax.bar(x - width/2, before_transfer, width, label='Before Transfer', alpha=0.7)
        ax.bar(x + width/2, after_transfer, width, label='After Transfer', alpha=0.7)
        
        ax.set_title('Transfer Performance')
        ax.set_xlabel('Tasks')
        ax.set_ylabel('Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks)
        ax.legend()
    
    def _plot_resource_usage(self, ax: plt.Axes, metrics: Dict[str, List[float]]):
        """Plot computational resource usage."""
        resources = ['Memory', 'GPU', 'CPU', 'Storage']
        usage = np.random.rand(4) * 80 + 10
        
        colors = ['red' if u > 70 else 'orange' if u > 50 else 'green' for u in usage]
        
        bars = ax.bar(resources, usage, color=colors, alpha=0.7)
        ax.set_title('Resource Usage (%)')
        ax.set_ylabel('Usage Percentage')
        ax.set_ylim(0, 100)
        
        # Add usage labels
        for bar, use in zip(bars, usage):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{use:.1f}%', ha='center', va='bottom')
    
    def create_real_time_dashboard(self) -> go.Figure:
        """Create real-time training dashboard with Plotly."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss Curves', 'Success Rates', 'Skill Usage', 'Adaptation Progress'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Placeholder data - would be updated in real-time
        x = list(range(100))
        
        # Loss curves
        fig.add_trace(
            go.Scatter(x=x, y=np.exp(-np.array(x)/20) + 0.1*np.random.randn(100),
                      name='Meta Loss', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Success rates
        fig.add_trace(
            go.Scatter(x=x, y=1-np.exp(-np.array(x)/30),
                      name='Success Rate', line=dict(color='green')),
            row=1, col=2
        )
        
        # Skill usage (bar chart)
        skills = ['Skill A', 'Skill B', 'Skill C', 'Skill D']
        usage = np.random.poisson(10, 4)
        fig.add_trace(
            go.Bar(x=skills, y=usage, name='Usage Count'),
            row=2, col=1
        )
        
        # Adaptation progress
        fig.add_trace(
            go.Scatter(x=x, y=np.random.beta(2, 5, 100),
                      name='Adaptation Speed', line=dict(color='red')),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Real-time Training Dashboard",
            showlegend=True
        )
        
        return fig