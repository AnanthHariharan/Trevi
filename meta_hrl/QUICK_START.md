# Meta-HRL Quick Start Guide 🚀

This guide will get your Meta-Learning Hierarchical Skill Acquisition system up and running in minutes.

## ✅ Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

## 🎯 Quick Demo

### 1. Run the Basic Demo
```bash
cd meta_hrl
python demo_usage.py
```

This will:
- Create sample skills and hierarchical policies
- Simulate meta-learning training
- Generate visualization plots
- Create interactive HTML files

**Generated Files:**
- `skill_library_overview.png` - Complete skill analysis
- `training_dashboard.png` - Training metrics visualization  
- `interactive_skill_network.html` - Interactive composition graph

## 🌐 Interactive Dashboard

### 2. Start the Backend API
```bash
cd meta_hrl
python start_dashboard.py
```

The FastAPI backend will start at: http://localhost:8000
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

### 3. Start the Frontend Dashboard
```bash
cd meta_hrl/frontend/frontend
npm install  # First time only
npm start
```

The React dashboard will open at: http://localhost:3000

## 📊 Dashboard Features

### Main Dashboard
- **Real-time Metrics**: Live training progress
- **System Status**: Connection and health monitoring
- **Key Performance Indicators**: Success rates, skill counts

### Skill Library
- **Interactive Browser**: Search and filter skills
- **Composition Graphs**: Visual skill relationships
- **Skill Details**: Individual performance metrics

### Hierarchical Policy
- **Policy Execution**: Real-time policy monitoring
- **Goal Evolution**: Dynamic goal tracking
- **Skill Sequences**: Step-by-step execution

### Training Dashboard
- **Live Metrics**: Real-time loss curves
- **Success Tracking**: Performance evolution
- **Resource Monitoring**: System usage

### Advanced Visualizations  
- **Skill Embeddings**: PCA and t-SNE projections
- **Clustering Analysis**: Automatic skill grouping
- **Dimensionality Analysis**: Variance explained

## 🔄 Real-time Features

The system supports live updates via WebSocket:

1. **Start Training**: Run `demo_usage.py` with dashboard running
2. **Watch Live**: Metrics update automatically in the web interface
3. **Interactive Control**: Start/stop training from the dashboard

## 🛠 Development

### Backend Development
```bash
cd meta_hrl/frontend/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend Development
```bash
cd meta_hrl/frontend/frontend
npm install
npm start
```

### Adding New Visualizations
1. Add API endpoints in `frontend/backend/main.py`
2. Create React components in `frontend/frontend/src/components/`
3. Update visualization utilities in `utils/visualization.py`

## 📁 Project Structure

```
meta_hrl/
├── demo_usage.py           # Complete demo script
├── start_dashboard.py      # Backend startup script
├── core/                   # Core meta-learning components
├── algorithms/             # MAML, Option-Critic, HAC
├── utils/                  # Visualization utilities
├── frontend/
│   ├── backend/           # FastAPI backend
│   └── frontend/          # React dashboard
└── README_VISUALIZATION.md # Detailed visualization guide
```

## 🎨 Customization

### Adding New Skills
```python
from core.skill import ParametricSkill

skill = ParametricSkill(
    skill_id="custom_skill",
    name="Custom Skill",
    input_dim=10,
    output_dim=4
)
skill_library.add_skill(skill)
```

### Custom Metrics
```python
import requests

metrics = {
    'custom_metric': 0.85,
    'training_step': 100
}

requests.post(
    'http://localhost:8000/api/training/update-metrics',
    json=metrics
)
```

## 🚨 Troubleshooting

### Backend Issues
- **Port 8000 in use**: Change port in `start_dashboard.py`
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **Import errors**: Ensure you're in the correct directory

### Frontend Issues
- **Module not found**: Run `npm install` in `frontend/frontend/`
- **Build errors**: Check Node.js version (14+ required)
- **API connection**: Verify backend is running on port 8000

### Demo Issues
- **No visualizations**: Install matplotlib: `pip install matplotlib seaborn`
- **Missing plots**: Check write permissions in current directory
- **Dashboard offline**: Normal when running demo without backend

## 🎯 Next Steps

1. **Explore the Code**: Check out the core algorithms in `algorithms/`
2. **Add Custom Environments**: Create new environments in `environments/`
3. **Extend Visualizations**: Add new plots in `utils/visualization.py`
4. **Scale Up**: Use the framework with real robotic tasks

## 📚 Additional Resources

- `README_VISUALIZATION.md` - Detailed visualization guide
- `frontend/README.md` - Frontend architecture
- API docs at http://localhost:8000/docs when backend is running

Enjoy exploring meta-learning for hierarchical skill acquisition! 🧠🤖