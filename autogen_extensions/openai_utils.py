import os
from openai import OpenAI
from dotenv import load_dotenv
from autogen_extensions.config_utils import ROOT_FOLDER


def get_openai_client() -> OpenAI:
    """Load the OpenAI API key from .env in ROOT_FOLDER or environment and return a configured OpenAI client."""
    load_dotenv(dotenv_path=os.path.join(ROOT_FOLDER, ".env"))
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env file.")
    return OpenAI(api_key=api_key)
