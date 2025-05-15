import json
import logging
from typing import Dict, List
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from load_openai import get_openai_config

def main():
    # Load configuration
    config = load_config()
    
    # Get OpenAI configuration
    openai_config = get_openai_config()
    
    # Configure agents with OpenAI settings
    llm_config = {
        "config_list": [openai_config],
        "cache_seed": 42
    }
    
    # ... rest of the existing code ... 