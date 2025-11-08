#!/usr/bin/env python3
"""
FastAPI Service for RAG Chatbot
RESTful API endpoint for querying the RAG knowledge base
"""

import os
import sys
from typing import Optional, List, Dict
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from rag_chatbot import RAGChatbot, create_chatbot


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Query request model"""
    question: str = Field(..., description="The question to ask")
    use_context: Optional[bool] = Field(True, description="Use conversation context")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")


class QueryResponse(BaseModel):
    """Query response model"""
    response: str = Field(..., description="The chatbot's response")
    metadata: Dict = Field(..., description="Query metadata (timing, chunks, etc.)")
    session_id: Optional[str] = Field(None, description="Session ID")


class HistoryResponse(BaseModel):
    """Conversation history response"""
    history: List[Dict] = Field(..., description="Conversation history")
    session_id: str = Field(..., description="Session ID")


class StatsResponse(BaseModel):
    """Statistics response"""
    total_queries: int
    successful_queries: int
    average_time: float
    total_time: float
    session_id: str


# Global chatbot instances (one per session)
chatbots: Dict[str, RAGChatbot] = {}
default_index_dir: Optional[str] = None
default_chat_model: str = "llama3.2:1b"


def get_or_create_chatbot(session_id: Optional[str] = None) -> RAGChatbot:
    """Get or create a chatbot instance for a session"""
    if default_index_dir is None:
        raise HTTPException(
            status_code=500,
            detail="Chatbot not initialized. Please set INDEX_DIR environment variable."
        )
    
    # Use default session if none provided
    session_id = session_id or "default"
    
    if session_id not in chatbots:
        chatbots[session_id] = create_chatbot(
            index_dir=default_index_dir,
            chat_model=default_chat_model,
            enable_context=True
        )
    
    return chatbots[session_id]


# FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="RESTful API for querying RAG knowledge base",
    version="1.0.0"
)

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup"""
    global default_index_dir, default_chat_model
    
    default_index_dir = os.getenv("INDEX_DIR")
    default_chat_model = os.getenv("CHAT_MODEL", "llama3.2:1b")
    
    if default_index_dir and os.path.exists(default_index_dir):
        print(f"✅ Chatbot initialized with index: {default_index_dir}")
        print(f"✅ Using model: {default_chat_model}")
    else:
        print("⚠️  INDEX_DIR not set or invalid. Chatbot will be created on first request.")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "/query": "POST - Query the knowledge base",
            "/history": "GET - Get conversation history",
            "/stats": "GET - Get conversation statistics",
            "/clear": "POST - Clear conversation history",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "index_dir": default_index_dir,
        "model": default_chat_model,
        "active_sessions": len(chatbots)
    }


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the RAG knowledge base
    
    - **question**: The question to ask
    - **use_context**: Whether to use conversation context (default: True)
    - **session_id**: Session ID for conversation tracking (optional)
    """
    try:
        chatbot = get_or_create_chatbot(request.session_id)
        response, metadata = chatbot.query(
            request.question,
            use_context=request.use_context,
            verbose=False
        )
        
        return QueryResponse(
            response=response,
            metadata=metadata,
            session_id=request.session_id or "default"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/history", response_model=HistoryResponse)
async def get_history(
    session_id: Optional[str] = Query(None, description="Session ID")
):
    """Get conversation history for a session"""
    try:
        chatbot = get_or_create_chatbot(session_id)
        history = chatbot.get_conversation_history()
        return HistoryResponse(
            history=history,
            session_id=session_id or "default"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_stats(
    session_id: Optional[str] = Query(None, description="Session ID")
):
    """Get conversation statistics for a session"""
    try:
        chatbot = get_or_create_chatbot(session_id)
        history = chatbot.get_conversation_history()
        
        if not history:
            return StatsResponse(
                total_queries=0,
                successful_queries=0,
                average_time=0.0,
                total_time=0.0,
                session_id=session_id or "default"
            )
        
        total_queries = len(history)
        total_time = sum(m.get("metadata", {}).get("total_time", 0) for m in history)
        avg_time = total_time / total_queries if total_queries > 0 else 0
        successful = sum(1 for m in history if m.get("metadata", {}).get("success", False))
        
        return StatsResponse(
            total_queries=total_queries,
            successful_queries=successful,
            average_time=avg_time,
            total_time=total_time,
            session_id=session_id or "default"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.post("/clear")
async def clear_history(
    session_id: Optional[str] = Query(None, description="Session ID")
):
    """Clear conversation history for a session"""
    try:
        chatbot = get_or_create_chatbot(session_id)
        chatbot.clear_context()
        return {
            "message": "Conversation history cleared",
            "session_id": session_id or "default"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")


def main():
    """Run the FastAPI server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Chatbot FastAPI Server")
    parser.add_argument(
        "--index-dir",
        help="Path to the FAISS index directory (or set INDEX_DIR env var)"
    )
    parser.add_argument(
        "--model",
        default="llama3.2:1b",
        help="Chat model to use (default: llama3.2:1b)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    args = parser.parse_args()
    
    # Set environment variables
    if args.index_dir:
        os.environ["INDEX_DIR"] = args.index_dir
    if args.model:
        os.environ["CHAT_MODEL"] = args.model
    
    # Validate index directory
    index_dir = args.index_dir or os.getenv("INDEX_DIR")
    if not index_dir or not os.path.exists(index_dir):
        print("❌ Index directory not found. Please specify --index-dir or set INDEX_DIR env var")
        return 1
    
    print(f"🚀 Starting RAG Chatbot API server...")
    print(f"   Index: {index_dir}")
    print(f"   Model: {args.model}")
    print(f"   URL: http://{args.host}:{args.port}")
    print(f"   Docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "chatbot_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    sys.exit(main())
