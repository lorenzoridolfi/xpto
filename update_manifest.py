#!/usr/bin/env python3

import argparse
import json
import sys
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """Save data to a JSON file with proper formatting."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error: Failed to write to {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

def merge_topics_entities(topics_file: str, entities_file: str) -> Dict[str, Any]:
    """Merge topics and entities into a single JSON structure."""
    topics = load_json_file(topics_file)
    entities = load_json_file(entities_file)
    
    return {
        "topics": topics,
        "entities": entities
    }

def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def file_status(file_path: str) -> str:
    """Check the status of a file."""
    if not os.path.exists(file_path):
        return "non-existent"
    if os.path.getsize(file_path) == 0:
        return "empty"
    return "okay"

def setup_agents(config: Dict[str, Any]) -> tuple[AssistantAgent, AssistantAgent, UserProxyAgent]:
    """Set up AutoGen agents using configuration from config file."""
    llm_config = config.get("llm_config", {})
    agents_config = config.get("agents", {})
    
    # Get LLM configurations
    creator_llm_config = llm_config.get("creator", {})
    validator_llm_config = llm_config.get("validator", {})
    manager_llm_config = llm_config.get("manager", {})
    
    # Get agent configurations
    creator_config = agents_config.get("creator", {})
    validator_config = agents_config.get("validator", {})
    
    # Create agents with their system messages and LLM configs
    creator = AssistantAgent(
        name=creator_config.get("name", "creator"),
        llm_config=creator_llm_config,
        system_message=creator_config.get("system_message", ""),
        description=creator_config.get("description", "Creator agent for generating initial metadata")
    )
    
    validator = AssistantAgent(
        name=validator_config.get("name", "validator"),
        llm_config=validator_llm_config,
        system_message=validator_config.get("system_message", ""),
        description=validator_config.get("description", "Validator agent for reviewing and improving metadata")
    )
    
    user_proxy = UserProxyAgent(
        name="user_proxy",
        code_execution_config={"use_docker": False}
    )
    
    return creator, validator, user_proxy

def validate_metadata(metadata: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Validate metadata against requirements."""
    validated = metadata.copy()
    
    # Validate summary
    summary_req = requirements.get("summary", {})
    if len(metadata.get("summary", "")) > summary_req.get("max_length", 200):
        validated["summary"] = metadata["summary"][:summary_req["max_length"]] + "..."
    
    # Validate keywords
    keywords_req = requirements.get("keywords", {})
    keywords = metadata.get("keywords", [])
    if len(keywords) < keywords_req.get("min_count", 3):
        keywords.extend(["keyword"] * (keywords_req["min_count"] - len(keywords)))
    elif len(keywords) > keywords_req.get("max_count", 10):
        keywords = keywords[:keywords_req["max_count"]]
    validated["keywords"] = [k.lower() for k in keywords]
    
    # Validate topics
    topics_req = requirements.get("topics", {})
    topics = metadata.get("topics", [])
    if len(topics) < topics_req.get("min_count", 1):
        topics.extend(["topic"] * (topics_req["min_count"] - len(topics)))
    elif len(topics) > topics_req.get("max_count", 5):
        topics = topics[:topics_req["max_count"]]
    validated["topics"] = topics
    
    # Validate entities
    entities_req = requirements.get("entities", {})
    entities = metadata.get("entities", [])
    if len(entities) < entities_req.get("min_count", 2):
        entities.extend(["entity"] * (entities_req["min_count"] - len(entities)))
    elif len(entities) > entities_req.get("max_count", 8):
        entities = entities[:entities_req["max_count"]]
    validated["entities"] = entities
    
    return validated

def run_reflection_agents(file_path: str, content: str, topics: Optional[List[str]], 
                         entities: Optional[List[str]], config: Dict[str, Any]) -> Dict:
    """Use AutoGen agents to reflect and generate metadata collaboratively."""
    topic_text = f"\nExpected topics: {', '.join(topics)}." if topics else ""
    entity_text = f"\nExpected entities: {', '.join(entities)}." if entities else ""

    # Get metadata requirements from config
    requirements = config.get("metadata_requirements", {})
    subject = config.get("subject", "")
    
    prompt_content = content
    user_prompt = (
        f"You are analyzing the content of the file: {file_path}\n"
        f"Subject: {subject}\n\n"
        f"Your goal is to generate metadata for the document including:\n"
        f"- summary (max {requirements.get('summary', {}).get('max_length', 200)} chars)\n"
        f"- keywords ({requirements.get('keywords', {}).get('min_count', 3)}-{requirements.get('keywords', {}).get('max_count', 10)} items)\n"
        f"- topics ({requirements.get('topics', {}).get('min_count', 1)}-{requirements.get('topics', {}).get('max_count', 5)} items)\n"
        f"- entities ({requirements.get('entities', {}).get('min_count', 2)}-{requirements.get('entities', {}).get('max_count', 8)} items)\n\n"
        f"Here is the content:\n{prompt_content}\n"
        f"{topic_text}\n{entity_text}"
    )

    creator, validator, user_proxy = setup_agents(config)
    
    group_chat = GroupChat(agents=[user_proxy, creator, validator], messages=[], max_round=3)
    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=config.get("llm_config", {}).get("manager", {})
    )

    reply = user_proxy.initiate_chat(
        manager,
        message=user_prompt,
        summary_method="reflection"
    )

    try:
        metadata = json.loads(reply)
        # Validate metadata against requirements
        validated_metadata = validate_metadata(metadata, requirements)
        return validated_metadata
    except:
        return {
            "summary": "N/A",
            "keywords": [],
            "topics": topics if topics else [],
            "entities": entities if entities else []
        }

def update_manifest(file_list: List[str], manifest: Dict[str, Any], 
                   topics_entities: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Update the manifest with file list and topics/entities information."""
    # Create a new manifest structure
    updated_manifest = {
        "files": [],
        "metadata": {
            "topics": topics_entities["topics"],
            "entities": topics_entities["entities"]
        }
    }
    
    # Get output format requirements
    output_format = config.get("output_format", {}).get("file_record", {})
    required_fields = output_format.get("required_fields", [])
    
    # Process each file
    for file_path in file_list:
        status = file_status(file_path)
        record = manifest.get(file_path, {})
        
        try:
            if status == "okay":
                file_hash = compute_file_hash(file_path)
                modified_date = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()

                if record.get("sha256") != file_hash:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    metadata = run_reflection_agents(
                        file_path, 
                        content, 
                        topics_entities["topics"].get("topics", []),
                        topics_entities["entities"].get("entities", []),
                        config
                    )
                    
                    file_record = {
                        "filename": str(Path(file_path).name),
                        "path": file_path,
                        "description": metadata.get("summary", f"Content from {Path(file_path).name}"),
                        "metadata": {
                            "summary": metadata.get("summary", ""),
                            "keywords": metadata.get("keywords", []),
                            "topics": metadata.get("topics", []),
                            "entities": metadata.get("entities", [])
                        },
                        "sha256": file_hash,
                        "modified_date": modified_date,
                        "status": status
                    }
                else:
                    file_record = record
            else:
                file_record = {
                    "filename": str(Path(file_path).name),
                    "path": file_path,
                    "description": f"File {Path(file_path).name} is {status}",
                    "status": status
                }
        except Exception as e:
            file_record = {
                "filename": str(Path(file_path).name),
                "path": file_path,
                "description": f"Error processing file: {str(e)}",
                "status": f"error: {str(e)}"
            }

        # Ensure all required fields are present
        for field in required_fields:
            if field not in file_record:
                file_record[field] = None

        updated_manifest["files"].append(file_record)
    
    return updated_manifest

def main():
    parser = argparse.ArgumentParser(
        description="Update manifest with file list and merge topics/entities"
    )
    parser.add_argument(
        "file_list",
        help="Path to file containing list of files to process"
    )
    parser.add_argument(
        "manifest",
        help="Path to existing manifest file (if any)"
    )
    parser.add_argument(
        "--topics",
        required=True,
        help="Path to topics JSON file"
    )
    parser.add_argument(
        "--entities",
        required=True,
        help="Path to entities JSON file"
    )
    parser.add_argument(
        "--config",
        default="update_manifest_config.json",
        help="Path to configuration file (default: update_manifest_config.json)"
    )
    parser.add_argument(
        "--output",
        default="updated_manifest.json",
        help="Output file path (default: updated_manifest.json)"
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not Path(args.file_list).exists():
        print(f"Error: File list not found: {args.file_list}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.topics).exists():
        print(f"Error: Topics file not found: {args.topics}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.entities).exists():
        print(f"Error: Entities file not found: {args.entities}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    # Load configuration
    config = load_json_file(args.config)
    
    # Extract commonly used config sections
    llm_config = config.get("llm_config", {})
    agents_config = config.get("agents", {})
    metadata_requirements = config.get("metadata_requirements", {})
    output_format = config.get("output_format", {}).get("file_record", {})
    required_fields = output_format.get("required_fields", [])
    
    # Load file list
    try:
        with open(args.file_list, 'r', encoding='utf-8') as f:
            file_list = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading file list: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load existing manifest if it exists
    manifest = {}
    if Path(args.manifest).exists():
        manifest = load_json_file(args.manifest)
    
    # Merge topics and entities
    topics_entities = merge_topics_entities(args.topics, args.entities)
    
    # Update manifest with the loaded configuration
    updated_manifest = update_manifest(
        file_list=file_list,
        manifest=manifest,
        topics_entities=topics_entities,
        config={
            "llm_config": llm_config,
            "agents": agents_config,
            "metadata_requirements": metadata_requirements,
            "output_format": {"file_record": {"required_fields": required_fields}}
        }
    )
    
    # Save updated manifest
    save_json_file(updated_manifest, args.output)
    print(f"Successfully updated manifest saved to: {args.output}")

if __name__ == "__main__":
    main()
