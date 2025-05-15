"""
FastAPI application for interfacing with the agent system.

This module provides a REST API interface for interacting with the agent system,
including authentication, question answering, feedback, and system reset capabilities.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import secrets
import logging
from datetime import datetime

from src.analytics_assistant_agent import AnalyticsAssistantAgent
from src.tool_analytics import ToolAnalytics
from src.llm_cache import LLMCache
from src.config import DEFAULT_CONFIG

# Initialize FastAPI app
app = FastAPI(
    title="Agent System API",
    description="API for interacting with the multi-agent system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic authentication
security = HTTPBasic()
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"  # In production, use environment variables

# Initialize agents and analytics
tool_analytics = ToolAnalytics()
llm_cache = LLMCache()
agent = AnalyticsAssistantAgent(
    name="analytics_assistant",
    description="Analytics assistant for processing questions and feedback",
    llm_config=DEFAULT_CONFIG.get("llm_config", {}),
    tool_analytics=tool_analytics,
    llm_cache=llm_cache
)

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    rationale: str
    critic: str
    timestamp: datetime

class FeedbackRequest(BaseModel):
    question_id: str
    feedback: str
    rating: Optional[int] = None

class FeedbackResponse(BaseModel):
    root_cause_analysis: Dict[str, Any]
    recommendations: list[str]
    timestamp: datetime

class ErrorResponse(BaseModel):
    detail: str

# Authentication dependency
def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.post("/login", response_model=Dict[str, str])
async def login(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Authenticate user with basic authentication.
    
    Returns:
        Dict with authentication status
    """
    get_current_user(credentials)
    return {"status": "authenticated", "message": "Login successful"}

@app.post("/question", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    username: str = Depends(get_current_user)
):
    """
    Ask a question to the agent system.
    
    Args:
        request: QuestionRequest containing the question
        
    Returns:
        QuestionResponse with answer, rationale, and critic
    """
    try:
        # Process the question
        response = await agent.run(request.question)
        
        # Generate rationale and critic
        rationale = await agent.run(f"Explain your reasoning for the answer: {response}")
        critic = await agent.run(f"Critically analyze this answer: {response}")
        
        return QuestionResponse(
            answer=response,
            rationale=rationale,
            critic=critic,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logging.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing question"
        )

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    username: str = Depends(get_current_user)
):
    """
    Submit feedback for a previous answer.
    
    Args:
        request: FeedbackRequest containing feedback and rating
        
    Returns:
        FeedbackResponse with root cause analysis and recommendations
    """
    try:
        # Process feedback
        feedback_prompt = f"""
        Analyze the following feedback:
        Question ID: {request.question_id}
        Feedback: {request.feedback}
        Rating: {request.rating if request.rating else 'Not provided'}
        
        Provide a root cause analysis and recommendations.
        """
        
        analysis = await agent.run(feedback_prompt)
        
        # Extract recommendations
        recommendations_prompt = f"Based on this analysis: {analysis}, provide specific recommendations for improvement."
        recommendations = await agent.run(recommendations_prompt)
        
        return FeedbackResponse(
            root_cause_analysis={"analysis": analysis},
            recommendations=[rec.strip() for rec in recommendations.split("\n") if rec.strip()],
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logging.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing feedback"
        )

@app.post("/reset", response_model=Dict[str, str])
async def reset_system(username: str = Depends(get_current_user)):
    """
    Reset the agent system's memory and cache.
    
    Returns:
        Dict with reset status
    """
    try:
        # Clear cache
        llm_cache.clear()
        
        # Reset analytics
        tool_analytics.reset()
        
        return {"status": "success", "message": "System reset successful"}
    except Exception as e:
        logging.error(f"Error resetting system: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error resetting system"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return ErrorResponse(detail=exc.detail)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 