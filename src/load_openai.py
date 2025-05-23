import os
from dotenv import load_dotenv
from typing import Optional


def load_openai_config() -> Optional[str]:
    """
    Load OpenAI API key from environment variables or .env file.
    Returns the API key if found, None otherwise.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Try to get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
            "or add it to your .env file."
        )

    return api_key


def get_openai_config() -> dict:
    """
    Get OpenAI configuration including API key and other settings.
    Returns a dictionary with the configuration.
    """
    api_key = load_openai_config()

    return {
        "api_key": api_key,
        "model": "gpt-4",  # Default model
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }
