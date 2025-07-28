# Meta-HRL Dashboard

Interactive web-based dashboard for visualizing and monitoring meta-learning hierarchical skill acquisition.

## Features

- **Real-time Monitoring**: Live updates via WebSocket connections
- **Skill Library Visualization**: Interactive skill composition graphs
- **Training Progress**: Real-time loss curves and metrics
- **Hierarchical Policy**: Policy execution monitoring
- **3D Visualizations**: Skill embeddings and environment rendering

## Setup

### Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend (React)

```bash
cd frontend
npm install
npm start
```

The dashboard will be available at `http://localhost:3000`

## Architecture

### Backend Components
- **FastAPI Server**: RESTful API and WebSocket endpoints
- **Real-time Updates**: WebSocket manager for live data streaming
- **Data Processing**: Skill library and training metrics APIs

### Frontend Components
- **React Dashboard**: Modern Material-UI interface
- **Real-time Charts**: Plotly.js for interactive visualizations
- **WebSocket Client**: Live updates without page refresh
- **Responsive Design**: Works on desktop and mobile

## API Endpoints

- `GET /api/skills` - Get all skills
- `GET /api/skill-library/stats` - Get library statistics
- `GET /api/composition-graph` - Get skill composition graph
- `GET /api/training/metrics` - Get training metrics
- `WS /ws` - WebSocket for real-time updates

## Usage

1. Start the backend server
2. Start the frontend development server
3. Navigate to the dashboard
4. Monitor training progress and skill development in real-time

## Screenshots

The dashboard includes:
- Overview dashboard with key metrics
- Skill library browser with search and filtering
- Interactive skill composition graphs
- Real-time training progress charts
- Hierarchical policy execution monitoring