"""
FastAPI application for interfacing with the agent system.

This module provides a REST API interface for interacting with the agent system,
including authentication, question answering, feedback, and system reset capabilities.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal
import secrets
import logging
from datetime import datetime, UTC
import uuid
import os
from pathlib import Path
from dotenv import load_dotenv
from autogen_extensions.log_utils import get_logger

# from src.analytics_assistant_agent import AnalyticsAssistantAgent
# from src.tool_analytics import ToolAnalytics
# from src.llm_cache import LLMCache
# from src.config import DEFAULT_CONFIG

# Load environment variables from parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize FastAPI app
app = FastAPI(
    title="Agent System API",
    description="API for interacting with the multi-agent system",
    version="1.0.0",
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
ADMIN_USERNAME = os.getenv("FASTAPI_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("FASTAPI_PASSWORD", "admin123")

# Initialize agents and analytics
# tool_analytics = ToolAnalytics()
# llm_cache = LLMCache()
# agent = AnalyticsAssistantAgent(
#     name="analytics_assistant",
#     description="Analytics assistant for processing questions and feedback",
#     llm_config=DEFAULT_CONFIG.get("llm_config", {}),
#     tool_analytics=tool_analytics,
#     llm_cache=llm_cache,
# )

# Store for question-answer pairs
question_store: Dict[str, Dict[str, Any]] = {}

# Question types
QuestionType = Literal["Synthetic User", "Personas"]

# Replace any instance of logging.getLogger(__name__) with get_logger(__name__)
logger = get_logger(__name__)


# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to ask the agent system")
    question_type: QuestionType = Field(
        ..., description="Type of question to help guide the response"
    )


class QuestionResponse(BaseModel):
    question_id: str = Field(
        ..., description="Unique identifier for the question-answer pair"
    )
    answer: str = Field(..., description="The answer provided by the agent")
    rationale: str = Field(..., description="The reasoning behind the answer")
    critic: str = Field(..., description="Critical analysis of the answer")
    timestamp: datetime = Field(..., description="When the response was generated")
    question_type: QuestionType = Field(
        ..., description="Type of question that was asked"
    )


class FeedbackRequest(BaseModel):
    question_id: str = Field(..., description="ID of the question being feedbacked")
    feedback: str = Field(..., description="User feedback about the answer")
    rating: Optional[int] = Field(
        None, ge=1, le=5, description="Optional rating from 1 to 5"
    )


class FeedbackResponse(BaseModel):
    root_cause_analysis: Dict[str, Any] = Field(
        ..., description="Analysis of the feedback"
    )
    recommendations: list[str] = Field(
        ..., description="List of improvement recommendations"
    )
    timestamp: datetime = Field(..., description="When the analysis was generated")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")


def generate_question_id() -> str:
    """Generate a unique question ID."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"q_{timestamp}_{unique_id}"


def validate_question_id(question_id: str) -> bool:
    """Validate if a question ID exists in the store."""
    return question_id in question_store


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
    request: QuestionRequest, username: str = Depends(get_current_user)
):
    """
    Ask a question to the agent system.

    Args:
        request: QuestionRequest containing the question and question type

    Returns:
        QuestionResponse with answer, rationale, and critic
    """
    try:
        # Generate unique question ID
        question_id = generate_question_id()

        # Prepare the question with type context
        question_prompt = f"[{request.question_type}] {request.question}"

        # Process the question
        # response = await agent.run(question_prompt)

        # Generate rationale and critic
        # rationale = await agent.run(
        #     f"Explain your reasoning for the answer: {response}"
        # )
        # critic = await agent.run(f"Critically analyze this answer: {response}")

        # Store the question-answer pair
        question_store[question_id] = {
            "question": request.question,
            "question_type": request.question_type,
            "answer": "",
            "rationale": "",
            "critic": "",
            "timestamp": datetime.now(UTC),
        }

        return QuestionResponse(
            question_id=question_id,
            answer="",
            rationale="",
            critic="",
            timestamp=datetime.now(UTC),
            question_type=request.question_type,
        )
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing question",
        )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest, username: str = Depends(get_current_user)
):
    """
    Submit feedback for a previous answer.

    Args:
        request: FeedbackRequest containing feedback and rating

    Returns:
        FeedbackResponse with root cause analysis and recommendations
    """
    try:
        # Validate question ID
        if not validate_question_id(request.question_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Question ID not found"
            )

        # Get the original question-answer pair
        qa_pair = question_store[request.question_id]

        # Process feedback
        feedback_prompt = f"""
        Analyze the following feedback:
        Question: {qa_pair['question']}
        Original Answer: {qa_pair['answer']}
        Feedback: {request.feedback}
        Rating: {request.rating if request.rating else 'Not provided'}
        
        Provide a root cause analysis and recommendations.
        """

        # analysis = await agent.run(feedback_prompt)

        # Extract recommendations
        # recommendations_prompt = f"Based on this analysis: {analysis}, provide specific recommendations for improvement."
        # recommendations = await agent.run(recommendations_prompt)

        return FeedbackResponse(
            root_cause_analysis={"analysis": ""},
            recommendations=[],
            timestamp=datetime.now(UTC),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing feedback",
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
        # llm_cache.clear()

        # Reset analytics
        # tool_analytics.reset()

        # Clear question store
        question_store.clear()

        return {"status": "success", "message": "System reset successful"}
    except Exception as e:
        logger.error(f"Error resetting system: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error resetting system",
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return ErrorResponse(detail=exc.detail)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
