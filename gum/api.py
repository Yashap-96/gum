# api.py - REST API for GUM server

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from .gum import gum
from .models import Observation, Proposition
from .db_utils import search_propositions_bm25
from .observers import Screen

# Pydantic models for API
class SuggestionItem(BaseModel):
    id: str
    title: str
    description: str
    benefit: int
    false_positive_cost: int
    false_negative_cost: int
    decay: int
    created_at: datetime
    last_updated: datetime
    status: str = "active"  # active, completed, dismissed

class PropositionItem(BaseModel):
    id: int
    text: str
    reasoning: str
    confidence: Optional[int]
    decay: Optional[int]
    support_score: float
    created_at: datetime
    updated_at: datetime

class ServerStatus(BaseModel):
    is_running: bool
    user_name: Optional[str]
    model: Optional[str]
    active_observers: List[str]
    last_activity: Optional[datetime]

class FeedbackRequest(BaseModel):
    suggestion_id: str
    feedback_type: str  # "thumbs_up", "thumbs_down", "complete", "dismiss"
    comment: Optional[str] = None

class StartRecordingRequest(BaseModel):
    user_name: str
    model: str = "gemini-2.5-flash"

class UpdatePropositionRequest(BaseModel):
    text: str
    reasoning: str

# Global GUM instance
gum_instance: Optional[gum] = None
gum_task: Optional[asyncio.Task] = None

app = FastAPI(title="GUM API", version="1.0.0")

# CORS middleware for web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001", 
        "http://localhost:5173", 
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:3001", 
        "http://127.0.0.1:5173"
    ],  # React dev servers
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# WebSocket connections for real-time updates
websocket_connections: List[WebSocket] = []

@app.on_event("startup")
async def startup_event():
    """Initialize the API server."""
    logging.info("GUM API server starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    global gum_instance, gum_task
    if gum_task:
        gum_task.cancel()
        try:
            await gum_task
        except asyncio.CancelledError:
            pass
    logging.info("GUM API server shutting down...")

async def broadcast_update(message: Dict[str, Any]):
    """Broadcast updates to all connected WebSocket clients."""
    if websocket_connections:
        await asyncio.gather(
            *[ws.send_text(json.dumps(message)) for ws in websocket_connections],
            return_exceptions=True
        )

@app.get("/")
async def root():
    """Root endpoint that redirects to API documentation."""
    return {
        "message": "GUM API Server",
        "version": "1.0.0",
        "endpoints": {
            "status": "/api/status",
            "recording": {
                "start": "/api/recording/start",
                "stop": "/api/recording/stop"
            },
            "suggestions": "/api/suggestions",
            "propositions": "/api/propositions",
            "websocket": "/ws"
        },
        "documentation": "/docs"
    }

@app.get("/api/status")
async def get_server_status() -> ServerStatus:
    """Get the current status of the GUM server."""
    global gum_instance
    
    if gum_instance is None:
        return ServerStatus(
            is_running=False,
            user_name=None,
            model=None,
            active_observers=[],
            last_activity=None
        )
    
    return ServerStatus(
        is_running=True,
        user_name=gum_instance.user_name,
        model=gum_instance.model,
        active_observers=[obs.name for obs in gum_instance.observers],
        last_activity=datetime.now(timezone.utc)
    )

@app.options("/api/recording/start")
async def options_start_recording():
    """Handle CORS preflight for start recording."""
    return {}

@app.post("/api/recording/start")
async def start_recording(request: StartRecordingRequest):
    """Start the GUM server with screen recording."""
    global gum_instance, gum_task
    
    if gum_instance is not None:
        raise HTTPException(status_code=400, detail="GUM server is already running")
    
    try:
        # Create and start GUM instance
        gum_instance = gum(request.user_name, request.model, Screen(request.model))
        await gum_instance.connect_db()
        gum_instance.start_update_loop()
        
        # Start the GUM server in background
        gum_task = asyncio.create_task(run_gum_server())
        
        await broadcast_update({
            "type": "server_status",
            "status": "started",
            "user_name": request.user_name,
            "model": request.model
        })
        
        return {"status": "started", "user_name": request.user_name, "model": request.model}
    
    except Exception as e:
        logging.error(f"Failed to start GUM server: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start server: {str(e)}")

@app.options("/api/recording/stop")
async def options_stop_recording():
    """Handle CORS preflight for stop recording."""
    return {}

@app.post("/api/recording/stop")
async def stop_recording():
    """Stop the GUM server."""
    global gum_instance, gum_task
    
    if gum_instance is None:
        raise HTTPException(status_code=400, detail="GUM server is not running")
    
    try:
        # Stop the GUM server
        if gum_task:
            gum_task.cancel()
            try:
                await gum_task
            except asyncio.CancelledError:
                pass
            gum_task = None
        
        # Clean up GUM instance
        await gum_instance.__aexit__(None, None, None)
        gum_instance = None
        
        await broadcast_update({
            "type": "server_status",
            "status": "stopped"
        })
        
        return {"status": "stopped"}
    
    except Exception as e:
        logging.error(f"Failed to stop GUM server: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop server: {str(e)}")

@app.get("/api/suggestions")
async def get_suggestions(limit: int = 10) -> List[SuggestionItem]:
    """Get AI-generated suggestions based on current GUM state."""
    global gum_instance
    
    if gum_instance is None:
        raise HTTPException(status_code=400, detail="GUM server is not running")
    
    try:
        # For now, generate basic suggestions from recent propositions
        async with gum_instance._session() as session:
            # Get recent propositions
            recent_props = await search_propositions_bm25(
                session, "", limit=limit, include_observations=False
            )
            
            suggestions = []
            for i, (prop, score) in enumerate(recent_props):
                # Convert proposition to suggestion (basic implementation)
                suggestion = SuggestionItem(
                    id=f"suggestion_{prop.id}",
                    title=f"Action based on: {prop.text[:50]}...",
                    description=prop.reasoning[:200] + "..." if len(prop.reasoning) > 200 else prop.reasoning,
                    benefit=prop.confidence or 5,
                    false_positive_cost=3,  # Default values for now
                    false_negative_cost=7,
                    decay=prop.decay or 5,
                    created_at=prop.created_at,
                    last_updated=prop.updated_at
                )
                suggestions.append(suggestion)
            
            return suggestions
    
    except Exception as e:
        logging.error(f"Failed to get suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")

@app.post("/api/suggestions/{suggestion_id}/feedback")
async def submit_suggestion_feedback(suggestion_id: str, feedback: FeedbackRequest):
    """Submit feedback for a suggestion."""
    # For now, just log the feedback
    logging.info(f"Feedback for suggestion {suggestion_id}: {feedback.feedback_type}")
    
    await broadcast_update({
        "type": "feedback_received",
        "suggestion_id": suggestion_id,
        "feedback_type": feedback.feedback_type
    })
    
    return {"status": "feedback_received"}

@app.get("/api/propositions")
async def get_propositions(limit: int = 50, search: Optional[str] = None) -> List[PropositionItem]:
    """Get propositions from the GUM."""
    global gum_instance
    
    if gum_instance is None:
        raise HTTPException(status_code=400, detail="GUM server is not running")
    
    try:
        async with gum_instance._session() as session:
            if search:
                props = await search_propositions_bm25(
                    session, search, limit=limit, include_observations=False
                )
            else:
                # Get recent propositions without search
                props = await search_propositions_bm25(
                    session, "", limit=limit, include_observations=False
                )
            
            proposition_items = []
            for prop, score in props:
                item = PropositionItem(
                    id=prop.id,
                    text=prop.text,
                    reasoning=prop.reasoning,
                    confidence=prop.confidence,
                    decay=prop.decay,
                    support_score=score,
                    created_at=prop.created_at,
                    updated_at=prop.updated_at
                )
                proposition_items.append(item)
            
            return proposition_items
    
    except Exception as e:
        logging.error(f"Failed to get propositions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get propositions: {str(e)}")

@app.put("/api/propositions/{proposition_id}")
async def update_proposition(proposition_id: int, request: UpdatePropositionRequest):
    """Update a proposition."""
    global gum_instance
    
    if gum_instance is None:
        raise HTTPException(status_code=400, detail="GUM server is not running")
    
    try:
        async with gum_instance._session() as session:
            # Get the proposition
            from sqlalchemy import select
            stmt = select(Proposition).where(Proposition.id == proposition_id)
            result = await session.execute(stmt)
            prop = result.scalar_one_or_none()
            
            if prop is None:
                raise HTTPException(status_code=404, detail="Proposition not found")
            
            # Update the proposition
            prop.text = request.text
            prop.reasoning = request.reasoning
            prop.updated_at = datetime.now(timezone.utc)
            
            await session.commit()
            
            await broadcast_update({
                "type": "proposition_updated",
                "proposition_id": proposition_id
            })
            
            return {"status": "updated"}
    
    except Exception as e:
        logging.error(f"Failed to update proposition: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update proposition: {str(e)}")

@app.delete("/api/propositions/{proposition_id}")
async def delete_proposition(proposition_id: int):
    """Delete a proposition."""
    global gum_instance
    
    if gum_instance is None:
        raise HTTPException(status_code=400, detail="GUM server is not running")
    
    try:
        async with gum_instance._session() as session:
            # Get the proposition
            from sqlalchemy import select
            stmt = select(Proposition).where(Proposition.id == proposition_id)
            result = await session.execute(stmt)
            prop = result.scalar_one_or_none()
            
            if prop is None:
                raise HTTPException(status_code=404, detail="Proposition not found")
            
            # Delete the proposition
            await session.delete(prop)
            await session.commit()
            
            await broadcast_update({
                "type": "proposition_deleted",
                "proposition_id": proposition_id
            })
            
            return {"status": "deleted"}
    
    except Exception as e:
        logging.error(f"Failed to delete proposition: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete proposition: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)

async def run_gum_server():
    """Run the GUM server in the background."""
    global gum_instance
    
    if gum_instance is None:
        return
    
    try:
        # Run the GUM server indefinitely
        await asyncio.Future()  # Run forever
    except asyncio.CancelledError:
        # Clean shutdown
        if gum_instance:
            await gum_instance.__aexit__(None, None, None)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 