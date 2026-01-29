from fastapi import WebSocket
from typing import Dict, List
import asyncio
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            if websocket in self.active_connections[job_id]:
                self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
    
    async def broadcast(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append(connection)
            
            for conn in disconnected:
                self.disconnect(conn, job_id)
    
    async def broadcast_all(self, message: dict):
        for job_id in list(self.active_connections.keys()):
            await self.broadcast(job_id, message)

manager = ConnectionManager()

async def send_training_update(
    job_id: str,
    status: str,
    progress: float,
    metrics: dict = None,
):
    message = {
        "type": "training_update",
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "metrics": metrics or {},
    }
    await manager.broadcast(job_id, message)

async def send_trajectory_update(
    job_id: str,
    trajectory_data: list,
    step: int,
):
    message = {
        "type": "trajectory_update",
        "job_id": job_id,
        "trajectory": trajectory_data,
        "step": step,
    }
    await manager.broadcast(job_id, message)

