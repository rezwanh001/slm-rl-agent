#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
Inference Server for SLM-RL-Agent

This module provides a REST API server for deploying trained SLM agents in production.
The server wraps your trained model and exposes it through HTTP endpoints, making it
easy to integrate with other applications and services.

WHY USE AN INFERENCE SERVER?
While you can use the SLMAgent directly in Python, a server offers several advantages:
1. Language-agnostic: Any application can call HTTP endpoints
2. Scalability: Run multiple model replicas behind a load balancer
3. Decoupling: Separate the ML model from the business logic
4. Monitoring: Track requests, latency, and errors centrally

ARCHITECTURE:
The server uses FastAPI for high performance async request handling. Each request:
1. Receives JSON with prompt and optional parameters
2. Validates inputs using Pydantic models
3. Generates a response using the SLMAgent
4. Returns the response as JSON

DEPLOYMENT OPTIONS:
- Development: `python -m src.agent.inference_server --model_path ./outputs/ppo/final`
- Production: Use uvicorn with gunicorn workers:
  `gunicorn -k uvicorn.workers.UvicornWorker -w 4 src.agent.inference_server:app`
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# Optional imports for server functionality
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from src.agent.slm_agent import SLMAgent, GenerationConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

if HAS_FASTAPI:
    class GenerateRequest(BaseModel):
        """
        Request model for the /generate endpoint.
        
        All fields except 'prompt' are optional and will use server defaults if not provided.
        """
        prompt: str = Field(..., description="The input text to generate a response for")
        max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
        temperature: Optional[float] = Field(None, description="Sampling temperature (0-2)")
        top_p: Optional[float] = Field(None, description="Nucleus sampling threshold")
        stop_sequences: Optional[List[str]] = Field(None, description="Sequences to stop on")
        stream: bool = Field(False, description="Stream the response (not yet implemented)")
        
        class Config:
            json_schema_extra = {
                "example": {
                    "prompt": "Explain machine learning in simple terms.",
                    "max_tokens": 200,
                    "temperature": 0.7
                }
            }
    
    class GenerateResponse(BaseModel):
        """Response model for the /generate endpoint."""
        response: str = Field(..., description="Generated text response")
        model: str = Field(..., description="Model identifier")
        usage: Dict[str, int] = Field(..., description="Token usage statistics")
        latency_ms: float = Field(..., description="Generation latency in milliseconds")
    
    class HealthResponse(BaseModel):
        """Response model for health check endpoint."""
        status: str
        model_loaded: bool
        model_info: Optional[Dict[str, Any]]
    
    class ChatMessage(BaseModel):
        """A single message in a conversation."""
        role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
        content: str = Field(..., description="Message content")
    
    class ChatRequest(BaseModel):
        """Request model for the /chat endpoint (multi-turn conversation)."""
        messages: List[ChatMessage] = Field(..., description="Conversation history")
        max_tokens: Optional[int] = Field(None)
        temperature: Optional[float] = Field(None)
        top_p: Optional[float] = Field(None)


class InferenceServer:
    """
    Production-ready inference server for SLM agents.
    
    This class manages the FastAPI application, model loading, and request handling.
    It provides a clean interface for both programmatic use and command-line deployment.
    
    Example:
        >>> # Programmatic usage
        >>> server = InferenceServer("./outputs/ppo/final")
        >>> server.start(host="0.0.0.0", port=8000)
        
        >>> # Or use the FastAPI app directly
        >>> from src.agent.inference_server import create_app
        >>> app = create_app("./outputs/ppo/final")
        >>> # Deploy with uvicorn or gunicorn
    """
    
    def __init__(
        self,
        model_path: str,
        generation_config: Optional[GenerationConfig] = None,
        max_concurrent_requests: int = 10,
    ):
        """
        Initialize the inference server.
        
        Args:
            model_path: Path to the trained model
            generation_config: Default generation settings
            max_concurrent_requests: Maximum concurrent requests to handle
        """
        if not HAS_FASTAPI:
            raise ImportError(
                "FastAPI is required for the inference server. "
                "Install with: pip install fastapi uvicorn"
            )
        
        self.model_path = model_path
        self.generation_config = generation_config
        self.max_concurrent_requests = max_concurrent_requests
        
        # Will be loaded when server starts
        self.agent: Optional[SLMAgent] = None
        self.app: Optional[FastAPI] = None
        
        # Request tracking
        self.request_count = 0
        self.total_latency = 0.0
    
    def load_model(self) -> None:
        """Load the model into memory."""
        logger.info(f"Loading model from {self.model_path}")
        
        self.agent = SLMAgent.from_pretrained(
            self.model_path,
            generation_config=self.generation_config,
        )
        
        logger.info("Model loaded successfully")
    
    def create_app(self) -> "FastAPI":
        """
        Create and configure the FastAPI application.
        
        Returns:
            Configured FastAPI app
        """
        app = FastAPI(
            title="SLM-RL-Agent Inference Server",
            description="REST API for generating text with trained Small Language Model agents",
            version="1.0.0",
        )
        
        # Add CORS middleware for browser access
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Store reference to server instance
        app.state.server = self
        
        # Register routes
        self._register_routes(app)
        
        self.app = app
        return app
    
    def _register_routes(self, app: FastAPI) -> None:
        """Register all API endpoints."""
        
        @app.on_event("startup")
        async def startup_event():
            """Load model when server starts."""
            self.load_model()
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """
            Health check endpoint.
            
            Use this endpoint for:
            - Kubernetes liveness/readiness probes
            - Load balancer health checks
            - Monitoring systems
            """
            return HealthResponse(
                status="healthy" if self.agent else "loading",
                model_loaded=self.agent is not None,
                model_info=self.agent.get_model_info() if self.agent else None,
            )
        
        @app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            """
            Generate text for a single prompt.
            
            This is the main endpoint for text generation. Send a prompt and
            receive a generated response.
            
            Example request:
            ```json
            {
                "prompt": "What is the capital of France?",
                "max_tokens": 100,
                "temperature": 0.7
            }
            ```
            """
            if self.agent is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            start_time = time.time()
            
            try:
                response_text = self.agent.generate(
                    prompt=request.prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop_sequences=request.stop_sequences,
                )
            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Update stats
            self.request_count += 1
            self.total_latency += latency_ms
            
            # Estimate token usage (rough approximation)
            input_tokens = len(request.prompt.split())
            output_tokens = len(response_text.split())
            
            return GenerateResponse(
                response=response_text,
                model=self.model_path,
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                latency_ms=latency_ms,
            )
        
        @app.post("/chat")
        async def chat(request: ChatRequest):
            """
            Chat endpoint for multi-turn conversations.
            
            Send the full conversation history and receive the assistant's next response.
            This is useful for building conversational interfaces.
            """
            if self.agent is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            # Build the full conversation context
            # For small models, we need to be careful about context length
            context_parts = []
            for msg in request.messages:
                role = msg.role.capitalize()
                context_parts.append(f"{role}: {msg.content}")
            
            # Add prompt for assistant response
            context_parts.append("Assistant:")
            full_prompt = "\n\n".join(context_parts)
            
            start_time = time.time()
            
            try:
                response_text = self.agent.generate(
                    prompt=full_prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                )
            except Exception as e:
                logger.error(f"Chat error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "response": response_text,
                "latency_ms": latency_ms,
            }
        
        @app.get("/stats")
        async def get_stats():
            """Get server statistics."""
            avg_latency = (
                self.total_latency / self.request_count
                if self.request_count > 0 else 0
            )
            
            return {
                "request_count": self.request_count,
                "average_latency_ms": avg_latency,
                "model_path": self.model_path,
            }
    
    def start(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
    ) -> None:
        """
        Start the inference server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
            reload: Enable auto-reload for development
        """
        if self.app is None:
            self.create_app()
        
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
        )


def create_app(model_path: str) -> "FastAPI":
    """
    Create a FastAPI app for the given model.
    
    This function is the entry point for production deployment with
    gunicorn or similar ASGI servers.
    
    Usage:
        gunicorn -k uvicorn.workers.UvicornWorker \\
            "src.agent.inference_server:create_app('./outputs/ppo/final')"
    """
    server = InferenceServer(model_path)
    return server.create_app()


def start_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    **kwargs,
) -> None:
    """
    Convenience function to start the inference server.
    
    Args:
        model_path: Path to the trained model
        host: Host to bind to
        port: Port to listen on
        **kwargs: Additional arguments passed to InferenceServer
    """
    server = InferenceServer(model_path, **kwargs)
    server.start(host=host, port=port)


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Command-line entry point for the inference server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SLM-RL-Agent Inference Server")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    
    args = parser.parse_args()
    
    server = InferenceServer(args.model_path)
    server.start(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
