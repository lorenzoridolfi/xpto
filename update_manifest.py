#!/usr/bin/env python3

import argparse
import json
import sys
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import jsonschema
import shutil
import logging
import glob

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from load_openai import get_openai_config

# Constants
MANIFEST_VERSION = "1.0.0"

def get_openai_api_key() -> str:
    """Get OpenAI API key from environment variable."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return api_key

def setup_logging(config: Dict) -> None:
    """Configure logging based on the configuration file."""
    log_config = config["logging"]
    handlers = []
    
    # Add file handler
    handlers.append(logging.FileHandler(log_config["file"]))
    
    # Add console handler if enabled
    if log_config.get("console", True):
        handlers.append(logging.StreamHandler())
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config["level"]),
        format=log_config["format"],
        handlers=handlers
    )

def load_json_file(file_path: str) -> Dict:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        raise

def save_json_file(data: Dict, file_path: str) -> None:
    """Save data to a JSON file with proper formatting."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {str(e)}")
        raise

def load_schema(config: Dict) -> Dict:
    """Load the manifest schema."""
    try:
        return load_json_file(config["system"]["schema_file"])
    except Exception as e:
        logger.error(f"Error loading schema: {str(e)}")
        raise

def validate_manifest(manifest: Dict, config: Dict) -> bool:
    """Validate manifest against schema."""
    try:
        schema = load_schema(config)
        jsonschema.validate(instance=manifest, schema=schema)
        return True
    except jsonschema.exceptions.ValidationError as e:
        logger.error(f"Manifest validation error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        return False

def create_backup(manifest_data: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Create a backup of the manifest file."""
    backup_config = config["system"]["backup"]
    if not backup_config["enabled"]:
        return ""

    backup_dir = backup_config["directory"]
    os.makedirs(backup_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime(backup_config["timestamp_format"])
    backup_filename = backup_config["filename_pattern"].format(timestamp=timestamp)
    backup_path = os.path.join(backup_dir, backup_filename)
    
    with open(backup_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    
    return backup_path

def cleanup_old_backups(config: Dict[str, Any]) -> None:
    """Remove old backup files keeping only the most recent ones."""
    backup_config = config["system"]["backup"]
    if not backup_config["enabled"]:
        return

    backup_dir = backup_config["directory"]
    if not os.path.exists(backup_dir):
        return

    backup_files = sorted(
        [f for f in os.listdir(backup_dir) if f.startswith("manifest_") and f.endswith(".json")],
        key=lambda x: os.path.getmtime(os.path.join(backup_dir, x)),
        reverse=True
    )

    max_backups = backup_config["max_count"]
    for old_file in backup_files[max_backups:]:
        try:
            os.remove(os.path.join(backup_dir, old_file))
            logger.debug(f"Removed old backup: {old_file}")
        except Exception as e:
            logger.error(f"Error removing old backup {old_file}: {e}")

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash: {str(e)}")
        raise

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get detailed information about a file."""
    try:
        stat = os.stat(file_path)
        return {
            "size": stat.st_size,
            "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "file_type": Path(file_path).suffix[1:] or "text",
            "encoding": "utf-8"  # Default encoding
        }
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        raise

def expand_glob_pattern(pattern: str, base_dir: str = "files") -> List[str]:
    """Expand a glob pattern into a list of matching files.
    
    Args:
        pattern (str): Glob pattern to expand
        base_dir (str): Base directory to search in
        
    Returns:
        List[str]: List of matching file paths
    """
    try:
        # Ensure the pattern is relative to the base directory
        if not pattern.startswith(base_dir):
            pattern = os.path.join(base_dir, pattern)
        
        # Expand the glob pattern
        matches = glob.glob(pattern, recursive=True)
        
        # Filter out directories
        matches = [m for m in matches if os.path.isfile(m)]
        
        logger.info(f"Expanded glob pattern '{pattern}' to {len(matches)} files")
        return matches
    except Exception as e:
        logger.error(f"Error expanding glob pattern '{pattern}': {str(e)}")
        raise

def process_files(config: Dict) -> Dict:
    """Process files and generate manifest with enhanced features."""
    try:
        # Initialize manifest structure
        manifest = {
            "version": MANIFEST_VERSION,
            "files": [],
            "metadata": {
                "topics": {},
                "entities": {},
                "statistics": {
                    "total_files": 0,
                    "total_size": 0,
                    "last_updated": datetime.now().isoformat()
                }
            }
        }
        
        # Process each file pattern
        for file_info in config["file_manifest"]:
            filename_pattern = file_info["filename"]
            description = file_info["description"]
            
            # Expand glob pattern
            matching_files = expand_glob_pattern(filename_pattern)
            
            if not matching_files:
                logger.warning(f"No files found matching pattern: {filename_pattern}")
                continue
            
            # Process each matching file
            for file_path in matching_files:
                # Get file status and info
                status = "okay"
                if not os.path.exists(file_path):
                    status = "non-existent"
                elif os.path.getsize(file_path) == 0:
                    status = "empty"
                
                # Get file details
                file_details = get_file_info(file_path) if status == "okay" else {}
                
                # Create file entry
                file_entry = {
                    "filename": os.path.basename(file_path),
                    "path": file_path,
                    "description": description,
                    "status": status,
                    "metadata": {
                        "summary": "",
                        "keywords": [],
                        "topics": [],
                        "entities": []
                    },
                    **file_details
                }
                
                # Add SHA-256 hash if file exists
                if status == "okay":
                    file_entry["sha256"] = calculate_file_hash(file_path)
                
                manifest["files"].append(file_entry)
                
                # Update statistics
                if status == "okay":
                    manifest["metadata"]["statistics"]["total_files"] += 1
                    manifest["metadata"]["statistics"]["total_size"] += file_details["size"]
        
        return manifest
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        raise

def create_agents(config: Dict) -> Dict[str, Any]:
    """Create and configure the agents using settings from config file."""
    logger.debug("Creating agents...")
    
    # Get OpenAI API key
    api_key = get_openai_api_key()
    
    # Update LLM configs with API key
    for agent_type in ["creator", "validator", "manager"]:
        config["llm_config"][agent_type]["api_key"] = api_key
    
    # Create Creator Agent
    creator_config = config["agents"]["creator"]
    creator = AssistantAgent(
        name=creator_config["name"],
        system_message=creator_config["system_message"],
        llm_config=config["llm_config"]["creator"]
    )
    logger.debug(f"Creator Agent created: {creator_config['description']}")
    
    # Create Critic Agent
    critic_config = config["agents"]["validator"]  # Using validator as critic
    critic = AssistantAgent(
        name=critic_config["name"],
        system_message=critic_config["system_message"],
        llm_config=config["llm_config"]["validator"]
    )
    logger.debug(f"Critic Agent created: {critic_config['description']}")
    
    # Create Manager Agent
    manager_config = config["agents"]["manager"]
    manager = AssistantAgent(
        name=manager_config["name"],
        system_message=manager_config["system_message"],
        llm_config=config["llm_config"]["manager"]
    )
    logger.debug(f"Manager Agent created: {manager_config['description']}")
    
    # Create User Proxy
    user_proxy_config = config["system"]["user_proxy"]
    user_proxy = UserProxyAgent(
        name=user_proxy_config["name"],
        human_input_mode=user_proxy_config["human_input_mode"],
        max_consecutive_auto_reply=user_proxy_config["max_consecutive_auto_reply"]
    )
    logger.debug("User Proxy Agent created")
    
    # Create Group Chat
    groupchat = GroupChat(
        agents=[user_proxy, creator, critic, manager],
        messages=[],
        max_round=config["system"]["group_chat"]["max_round"]
    )
    logger.debug("Group Chat created")
    
    # Create Group Chat Manager
    group_chat_manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=config["llm_config"]["manager"]
    )
    logger.debug("Group Chat Manager created")
    
    return {
        "creator": creator,
        "critic": critic,
        "manager": manager,
        "user_proxy": user_proxy,
        "group_chat_manager": group_chat_manager
    }

def main():
    """Main function to update the manifest."""
    try:
        # Load configuration
        config = load_json_file("update_manifest_config.json")
        logger.debug("Configuration loaded")
        
        # Setup logging based on config
        setup_logging(config)
        logger.debug("Logging configured")
        
        # Get OpenAI configuration
        openai_config = get_openai_config()
        
        # Configure agents with OpenAI settings
        llm_config = {
            "config_list": [openai_config],
            "cache_seed": 42
        }
        
        # Create agents
        agents = create_agents(config)
        logger.debug("Agents created and configured")
        
        # Initialize the process
        agents["user_proxy"].initiate_chat(
            agents["group_chat_manager"],
            message="""Let's create and validate a manifest for our files.
            The Manager should coordinate the process:
            1. Creator should analyze the files and generate the manifest
            2. Critic should review and provide feedback
            3. Manager should ensure the process completes successfully"""
        )
        
        # Process files and generate manifest
        manifest = process_files(config)
        logger.debug("Files processed and manifest generated")
        
        # Validate manifest
        if not validate_manifest(manifest, config):
            raise ValueError("Generated manifest failed validation")
        logger.debug("Manifest validated successfully")
        
        # Create backup before processing
        backup_path = create_backup(manifest, config)
        if backup_path:
            logger.info(f"Created backup at: {backup_path}")
        
        # Save new manifest
        save_json_file(manifest, config["system"]["manifest_file"])
        logger.debug("New manifest saved successfully")
        
        # Cleanup old backups
        cleanup_old_backups(config)
        
        logger.info("Manifest update completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
