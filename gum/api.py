# api.py - REST API for GUM server

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .gum import gum
from .models import Observation, Proposition
from .db_utils import search_propositions_bm25
from .observers import Screen

# Pydantic models for API
class SuggestionItem(BaseModel):
    id: int
    text: str
    reasoning: str
    utility_score: Optional[float]
    benefit: Optional[float]
    cost: Optional[float]
    status: str
    created_at: datetime
    updated_at: datetime

class GroupedSuggestionItem(BaseModel):
    id: int
    text: str
    reasoning: str
    utility_score: Optional[float]
    benefit: Optional[float]
    cost: Optional[float]
    status: str
    created_at: datetime
    updated_at: datetime
    topic: str
    topic_keywords: List[str]
    similar_count: int

class SuggestionGroup(BaseModel):
    topic: str
    topic_keywords: List[str]
    suggestions: List[GroupedSuggestionItem]
    total_count: int
    max_utility_score: float

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
    suggestion_id: int
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

def extract_topic_keywords(text: str, reasoning: str) -> List[str]:
    """Extract key topic keywords from suggestion text and reasoning."""
    # Combine text and reasoning
    combined_text = f"{text} {reasoning}".lower()
    
    # Common development-related keywords
    dev_keywords = [
        'gumbo', 'gum', 'development', 'coding', 'programming', 'debugging',
        'frontend', 'backend', 'api', 'database', 'server', 'client',
        'react', 'typescript', 'python', 'fastapi', 'sqlite', 'websocket',
        'screenshot', 'recording', 'observation', 'proposition', 'suggestion',
        'interface', 'ui', 'ux', 'component', 'hook', 'state', 'effect',
        'testing', 'deployment', 'git', 'version', 'control', 'merge',
        'error', 'bug', 'fix', 'optimize', 'performance', 'memory',
        'ai', 'machine learning', 'model', 'gemini', 'openai', 'api key'
    ]
    
    # Extract keywords that appear in the text
    found_keywords = []
    for keyword in dev_keywords:
        if keyword in combined_text:
            found_keywords.append(keyword)
    
    # Also extract any technical terms (words with camelCase, snake_case, or containing numbers)
    technical_terms = re.findall(r'\b[a-zA-Z]+[A-Z][a-zA-Z]*\b|\b[a-z]+_[a-z]+\b|\b\w*\d+\w*\b', combined_text)
    found_keywords.extend(technical_terms[:3])  # Limit to top 3 technical terms
    
    return list(set(found_keywords))[:5]  # Return unique keywords, max 5

def calculate_similarity(text1: str, text2: str, reasoning1: str, reasoning2: str) -> float:
    """Calculate similarity between two suggestions using TF-IDF and cosine similarity."""
    try:
        # Combine text and reasoning for each suggestion
        combined1 = f"{text1} {reasoning1}"
        combined2 = f"{text2} {reasoning2}"
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform([combined1, combined2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity
    except Exception as e:
        logging.error(f"Error calculating similarity: {e}")
        return 0.0

def group_similar_suggestions(suggestions: List[SuggestionItem], similarity_threshold: float = 0.7) -> List[SuggestionGroup]:
    """Group suggestions by similarity and topic."""
    if not suggestions:
        return []
    
    # Extract topics and keywords for each suggestion
    suggestions_with_topics = []
    for suggestion in suggestions:
        keywords = extract_topic_keywords(suggestion.text, suggestion.reasoning)
        # Create a topic name from the most prominent keywords
        topic = " ".join(keywords[:2]) if keywords else "General"
        
        suggestions_with_topics.append({
            'suggestion': suggestion,
            'topic': topic,
            'keywords': keywords,
            'grouped': False
        })
    
    # Group similar suggestions
    groups = []
    
    for i, item1 in enumerate(suggestions_with_topics):
        if item1['grouped']:
            continue
            
        # Start a new group
        current_group = [item1]
        item1['grouped'] = True
        
        # Find similar suggestions
        for j, item2 in enumerate(suggestions_with_topics[i+1:], i+1):
            if item2['grouped']:
                continue
                
            similarity = calculate_similarity(
                item1['suggestion'].text, item2['suggestion'].text,
                item1['suggestion'].reasoning, item2['suggestion'].reasoning
            )
            
            if similarity >= similarity_threshold:
                current_group.append(item2)
                item2['grouped'] = True
        
        # Create group with top 2 suggestions (max 2 per topic)
        current_group.sort(key=lambda x: x['suggestion'].utility_score or 0, reverse=True)
        top_suggestions = current_group[:2]
        
        # Create GroupedSuggestionItem for each suggestion
        grouped_suggestions = []
        for idx, item in enumerate(top_suggestions):
            grouped_suggestion = GroupedSuggestionItem(
                id=item['suggestion'].id,
                text=item['suggestion'].text,
                reasoning=item['suggestion'].reasoning,
                utility_score=item['suggestion'].utility_score,
                benefit=item['suggestion'].benefit,
                cost=item['suggestion'].cost,
                status=item['suggestion'].status,
                created_at=item['suggestion'].created_at,
                updated_at=item['suggestion'].updated_at,
                topic=item['topic'],
                topic_keywords=item['keywords'],
                similar_count=len(current_group)
            )
            grouped_suggestions.append(grouped_suggestion)
        
        # Create SuggestionGroup
        group = SuggestionGroup(
            topic=top_suggestions[0]['topic'],
            topic_keywords=list(set([kw for item in top_suggestions for kw in item['keywords']])),
            suggestions=grouped_suggestions,
            total_count=len(current_group),
            max_utility_score=max(item['suggestion'].utility_score or 0 for item in top_suggestions)
        )
        
        groups.append(group)
    
    # Sort groups by max utility score
    groups.sort(key=lambda x: x.max_utility_score, reverse=True)
    
    return groups

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
async def get_suggestions(
    limit: int = 10, 
    status: Optional[str] = None, 
) -> List[SuggestionItem]:
    """Get AI-generated suggestions based on current GUM state."""
    global gum_instance
    
    if gum_instance is None:
        raise HTTPException(status_code=400, detail="GUM server is not running")
    
    try:
        async with gum_instance._session() as session:
            stmt = select(Proposition).order_by(Proposition.utility_score.desc())
            if status:
                stmt = stmt.where(Proposition.status == status)
            if limit > 0:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            props = result.scalars().all()
            
            suggestions = [
                SuggestionItem(
                    id=prop.id,
                    text=prop.text,
                    reasoning=prop.reasoning,
                    utility_score=prop.utility_score,
                    benefit=prop.benefit,
                    cost=prop.cost,
                    status=prop.status,
                    created_at=prop.created_at,
                    updated_at=prop.updated_at,
                )
                for prop in props
            ]
            return suggestions
    
    except Exception as e:
        logging.error(f"Failed to get suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")

@app.get("/api/suggestions/grouped")
async def get_grouped_suggestions(
    limit: int = 10, 
    status: Optional[str] = None,
    similarity_threshold: float = 0.7
) -> List[SuggestionGroup]:
    """Get AI-generated suggestions grouped by topic and similarity."""
    global gum_instance
    
    if gum_instance is None:
        raise HTTPException(status_code=400, detail="GUM server is not running")
    
    try:
        async with gum_instance._session() as session:
            stmt = select(Proposition).order_by(Proposition.utility_score.desc())
            if status:
                stmt = stmt.where(Proposition.status == status)
            if limit > 0:
                stmt = stmt.limit(limit * 2)  # Get more to allow for grouping

            result = await session.execute(stmt)
            props = result.scalars().all()
            
            # Convert to SuggestionItem format
            suggestions = [
                SuggestionItem(
                    id=prop.id,
                    text=prop.text,
                    reasoning=prop.reasoning,
                    utility_score=prop.utility_score,
                    benefit=prop.benefit,
                    cost=prop.cost,
                    status=prop.status,
                    created_at=prop.created_at,
                    updated_at=prop.updated_at,
                )
                for prop in props
            ]
            
            # Group similar suggestions
            grouped_suggestions = group_similar_suggestions(suggestions, similarity_threshold)
            
            # Limit the number of groups returned
            return grouped_suggestions[:limit]
    
    except Exception as e:
        logging.error(f"Failed to get grouped suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get grouped suggestions: {str(e)}")

@app.post("/api/suggestions/feedback")
async def submit_suggestion_feedback(feedback: FeedbackRequest):
    """Submit feedback for a suggestion."""
    global gum_instance

    if gum_instance is None:
        raise HTTPException(status_code=400, detail="GUM server is not running")

    try:
        async with gum_instance._session() as session:
            stmt = select(Proposition).where(Proposition.id == feedback.suggestion_id)
            result = await session.execute(stmt)
            prop = result.scalar_one_or_none()

            if prop is None:
                raise HTTPException(status_code=404, detail="Suggestion not found")

            if feedback.feedback_type == "thumbs_up":
                # You might want to adjust benefit/cost here
                prop.benefit = (prop.benefit or 0) + 1
            elif feedback.feedback_type == "thumbs_down":
                prop.cost = (prop.cost or 0) + 1
            elif feedback.feedback_type == "complete":
                prop.status = "completed"
            elif feedback.feedback_type == "dismiss":
                prop.status = "dismissed"
            
            # Recalculate utility score
            if prop.benefit is not None and prop.cost is not None:
                prop.utility_score = prop.benefit - prop.cost
                
            prop.updated_at = datetime.now(timezone.utc)
            await session.commit()

            await broadcast_update({
                "type": "suggestion_updated",
                "suggestion": SuggestionItem.from_orm(prop).dict()
            })
            
            return {"status": "feedback_received"}
    except Exception as e:
        logging.error(f"Failed to process feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process feedback: {str(e)}")


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
            
            # Broadcast real-time update with full proposition data
            updated_item = PropositionItem(
                id=prop.id,
                text=prop.text,
                reasoning=prop.reasoning,
                confidence=prop.confidence,
                decay=prop.decay,
                support_score=0,  # Optionally recalculate if needed
                created_at=prop.created_at,
                updated_at=prop.updated_at
            )
            await broadcast_update({
                "type": "proposition_updated",
                "proposition": updated_item.dict()
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
            stmt = select(Proposition).where(Proposition.id == proposition_id)
            result = await session.execute(stmt)
            prop = result.scalar_one_or_none()
            
            if prop is None:
                raise HTTPException(status_code=404, detail="Proposition not found")
            
            # Delete the proposition
            await session.delete(prop)
            await session.commit()
            
            # Broadcast real-time delete event
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
