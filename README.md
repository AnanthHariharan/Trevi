# Meta-Learning for Hierarchical Skill Acquisition and Composition

A research framework for combining meta-learning with hierarchical reinforcement learning to learn reusable skills that can be quickly composed for complex, long-horizon tasks.

## Overview

This project implements a novel approach that:
- **Meta-learns primitive skills** using MAML and related techniques
- **Composes skills hierarchically** for complex task solving  
- **Transfers learned skills** efficiently to new domains
- **Discovers skill patterns** automatically from successful task executions

## Key Features

- **Multiple Meta-Learning Algorithms**: MAML, Option-Critic, HAC
- **Hierarchical Policy Architecture**: Multi-level decision making
- **Skill Composition Engine**: Automatic skill sequencing and optimization
- **Flexible Environments**: Robotics, navigation, multi-stage games
- **Comprehensive Evaluation**: Transfer efficiency, composability metrics

## Installation

```bash
# Basic installation
pip install -e .

# With full dependencies
pip install -e .[full]

# Development setup
pip install -e .[dev]
```

## Quick Start

```python
from meta_hrl.core import SkillLibrary, HierarchicalPolicy
from meta_hrl.algorithms.maml import SkillMAML
from meta_hrl.training import MetaTrainer

# Initialize components
skill_library = SkillLibrary()
meta_learner = SkillMAML(base_skill_config={
    'input_dim': 10,
    'output_dim': 4
})

# Train on multiple tasks
trainer = MetaTrainer(meta_learner, skill_library)
trainer.train(task_distribution, num_iterations=1000)

# Adapt to new task
new_skill = meta_learner.learn_new_skill(new_task_data)
```

## Architecture

### Core Components
- **Skill Library**: Manages learned primitive skills
- **Meta-Learner**: Learns skill initialization and adaptation
- **Skill Composer**: Sequences skills for complex tasks  
- **Hierarchical Policy**: Multi-level decision making

### Algorithms
- **MAML**: Model-agnostic meta-learning for skills
- **Option-Critic**: Hierarchical option learning
- **HAC**: Hindsight action critic
- **Compositional**: Skill discovery and adaptation

## Environments

- **Robotic Tasks**: Assembly, manipulation
- **Navigation**: Multi-room, continuous spaces
- **Games**: Multi-stage levels

## Evaluation Metrics

- **Transfer Efficiency**: Learning speed on new tasks
- **Skill Transferability**: Cross-domain skill reuse
- **Composition Quality**: Hierarchical structure interpretability
- **Sample Efficiency**: Data requirements for adaptation

## Research Applications

- Robotic assembly with transferable manipulation primitives
- Navigation in varied environments using composable movement skills  
- Multi-stage game solving with hierarchical strategy skills

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

```bibtex
@misc{meta-hierarchical-skills,
  title={Meta-Learning for Hierarchical Skill Acquisition and Composition},
  author={Research Team},
  year={2024},
  url={https://github.com/user/meta-hrl}
}
```