from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
import asyncio
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from datetime import datetime
import logging

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from core.skill import SkillLibrary, Skill
from core.hierarchical_policy import HierarchicalPolicy
from utils.visualization import SkillVisualization, HierarchicalPolicyVisualization, TrainingVisualization

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Meta-HRL Dashboard", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
skill_library = SkillLibrary()
hierarchical_policy = None
training_metrics = {}
connected_clients: List[WebSocket] = []

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if self.active_connections:
            message_str = json.dumps(message, default=self._json_serializer)
            for connection in self.active_connections:
                try:
                    await connection.send_text(message_str)
                except Exception as e:
                    logger.error(f"Error sending message to client: {e}")
    
    @staticmethod
    def _json_serializer(obj):
        """JSON serializer for non-serializable objects."""
        if torch.is_tensor(obj):
            return obj.cpu().numpy().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

manager = ConnectionManager()

@app.get("/")
async def read_root():
    """Serve the main dashboard."""
    return {"message": "Meta-HRL Dashboard API", "version": "1.0.0"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "skills_count": len(skill_library),
        "connections": len(manager.active_connections)
    }

@app.get("/api/skills")
async def get_skills():
    """Get all skills in the library."""
    skills_data = []
    
    for skill_id, skill in skill_library.skills.items():
        skill_data = {
            "id": skill_id,
            "name": skill.name,
            "input_dim": skill.input_dim,
            "output_dim": skill.output_dim,
            "success_rate": skill.success_rate,
            "is_trained": skill.is_trained
        }
        skills_data.append(skill_data)
    
    return {"skills": skills_data, "total_count": len(skills_data)}

@app.get("/api/skills/{skill_id}")
async def get_skill_details(skill_id: str):
    """Get detailed information about a specific skill."""
    skill = skill_library.get_skill(skill_id)
    
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    
    # Get skill embedding if available
    embedding = skill_library.skill_embeddings.get(skill_id)
    
    # Get composition information
    composition_chain = skill_library.get_composition_chain(skill_id)
    
    return {
        "id": skill_id,
        "name": skill.name,
        "input_dim": skill.input_dim,
        "output_dim": skill.output_dim,
        "success_rate": skill.success_rate,
        "is_trained": skill.is_trained,
        "embedding": embedding.tolist() if embedding is not None else None,
        "composition_chain": composition_chain,
        "is_composite": len(composition_chain) > 0
    }

@app.post("/api/skills")
async def create_skill(skill_data: Dict[str, Any]):
    """Create a new skill."""
    from ...core.skill import ParametricSkill
    
    try:
        new_skill = ParametricSkill(
            skill_id=skill_data["id"],
            name=skill_data["name"],
            input_dim=skill_data["input_dim"],
            output_dim=skill_data["output_dim"]
        )
        
        skill_library.add_skill(new_skill)
        
        # Broadcast update to connected clients
        await manager.broadcast({
            "type": "skill_created",
            "data": {
                "id": new_skill.skill_id,
                "name": new_skill.name,
                "timestamp": datetime.now().isoformat()
            }
        })
        
        return {"message": "Skill created successfully", "skill_id": new_skill.skill_id}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/skill-library/stats")
async def get_skill_library_stats():
    """Get skill library statistics."""
    skills = list(skill_library.skills.values())
    
    if not skills:
        return {
            "total_skills": 0,
            "avg_success_rate": 0,
            "trained_skills": 0,
            "composite_skills": 0
        }
    
    success_rates = [skill.success_rate for skill in skills]
    trained_count = sum(1 for skill in skills if skill.is_trained)
    composite_count = len([k for k, v in skill_library.composition_graph.items() if v])
    
    return {
        "total_skills": len(skills),
        "avg_success_rate": np.mean(success_rates),
        "min_success_rate": np.min(success_rates),
        "max_success_rate": np.max(success_rates),
        "trained_skills": trained_count,
        "composite_skills": composite_count,
        "skill_dimensions": {
            "avg_input_dim": np.mean([skill.input_dim for skill in skills]),
            "avg_output_dim": np.mean([skill.output_dim for skill in skills])
        }
    }

@app.get("/api/skill-library/similarity-matrix")
async def get_skill_similarity_matrix():
    """Get skill similarity matrix."""
    skills = list(skill_library.skills.keys())
    n_skills = len(skills)
    
    if n_skills < 2:
        return {"matrix": [], "skills": skills}
    
    similarity_matrix = []
    
    for i, skill1 in enumerate(skills):
        row = []
        for j, skill2 in enumerate(skills):
            if i == j:
                similarity = 1.0
            else:
                similarity = skill_library.get_skill_similarity(skill1, skill2)
            row.append(similarity)
        similarity_matrix.append(row)
    
    return {
        "matrix": similarity_matrix,
        "skills": skills,
        "dimensions": [n_skills, n_skills]
    }

@app.get("/api/composition-graph")
async def get_composition_graph():
    """Get skill composition graph data."""
    nodes = []
    edges = []
    
    # Add all skills as nodes
    for skill_id, skill in skill_library.skills.items():
        nodes.append({
            "id": skill_id,
            "name": skill.name,
            "success_rate": skill.success_rate,
            "is_composite": skill_id in skill_library.composition_graph and 
                           len(skill_library.composition_graph[skill_id]) > 0
        })
    
    # Add composition relationships as edges
    for composite_id, component_ids in skill_library.composition_graph.items():
        for component_id in component_ids:
            if component_id in skill_library.skills:
                edges.append({
                    "source": component_id,
                    "target": composite_id,
                    "relationship": "composes"
                })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges)
    }

@app.get("/api/training/metrics")
async def get_training_metrics():
    """Get current training metrics."""
    return {
        "metrics": training_metrics,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/training/update-metrics")
async def update_training_metrics(metrics: Dict[str, Any]):
    """Update training metrics."""
    global training_metrics
    
    # Update metrics
    for key, value in metrics.items():
        if key not in training_metrics:
            training_metrics[key] = []
        training_metrics[key].append(value)
        
        # Keep only last 1000 values
        if len(training_metrics[key]) > 1000:
            training_metrics[key] = training_metrics[key][-1000:]
    
    # Broadcast update to connected clients
    await manager.broadcast({
        "type": "metrics_updated",
        "data": {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    })
    
    return {"message": "Metrics updated successfully"}

@app.get("/api/hierarchy/status")
async def get_hierarchy_status():
    """Get current hierarchical policy status."""
    if hierarchical_policy is None:
        return {"status": "not_initialized"}
    
    hierarchy_state = hierarchical_policy.get_current_hierarchy_state()
    
    return {
        "status": "active",
        "current_goal": hierarchy_state.get("current_goal"),
        "skill_sequence": hierarchy_state.get("skill_sequence", []),
        "current_skill_index": hierarchy_state.get("skill_index", 0),
        "current_skill": hierarchy_state.get("current_skill"),
        "sequence_progress": hierarchy_state.get("sequence_progress", 0),
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    websocket
                )
            elif message.get("type") == "subscribe":
                # Handle subscription to specific data streams
                await manager.send_personal_message(
                    json.dumps({
                        "type": "subscribed",
                        "channels": message.get("channels", [])
                    }),
                    websocket
                )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Background task for periodic updates
async def periodic_updates():
    """Send periodic updates to connected clients."""
    while True:
        if manager.active_connections:
            # Send skill library stats
            try:
                stats = await get_skill_library_stats()
                await manager.broadcast({
                    "type": "stats_update",
                    "data": stats
                })
            except Exception as e:
                logger.error(f"Error in periodic updates: {e}")
        
        await asyncio.sleep(5)  # Update every 5 seconds

@app.on_event("startup")
async def startup_event():
    """Start background tasks."""
    # Start periodic updates
    asyncio.create_task(periodic_updates())
    logger.info("Meta-HRL Dashboard API started")

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Meta-HRL Dashboard API shutting down")

# Utility endpoints for visualization data
@app.get("/api/visualization/skill-embeddings")
async def get_skill_embeddings_visualization():
    """Get skill embeddings for 2D/3D visualization."""
    if not skill_library.skill_embeddings:
        return {"embeddings": [], "labels": []}
    
    embeddings = []
    labels = []
    
    for skill_id, embedding in skill_library.skill_embeddings.items():
        embeddings.append(embedding.cpu().numpy().tolist())
        labels.append(skill_id)
    
    # Perform dimensionality reduction for visualization
    if len(embeddings) > 1:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        embeddings_array = np.array(embeddings)
        
        # PCA to 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_array)
        
        # t-SNE to 2D (if we have enough points)
        embeddings_tsne = None
        if len(embeddings) > 3:
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_tsne = tsne.fit_transform(embeddings_array)
        
        return {
            "embeddings_raw": embeddings,
            "embeddings_2d_pca": embeddings_2d.tolist(),
            "embeddings_2d_tsne": embeddings_tsne.tolist() if embeddings_tsne is not None else None,
            "labels": labels,
            "pca_explained_variance": pca.explained_variance_ratio_.tolist()
        }
    
    return {"embeddings": embeddings, "labels": labels}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)