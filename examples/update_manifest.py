"""
Update Manifest Example

This example demonstrates a multi-agent system that:
1. Generates a manifest for project files
2. Validates the manifest against a schema
3. Saves the validated manifest
4. Collects a trace of all agent interactions

Uses autogen's GroupChat for orchestration and real agent implementations.
"""

import os
import logging
import json
import hashlib
import datetime
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
import openai
from autogen import GroupChat

from examples.common import (
    load_json_file,
    save_json_file,
    JsonSchemaValidationError,
    read_file_content,
    validate_manifest,
)

# Set up logging before any logger.info calls
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = "/Users/lorenzo/Sync/Source/AI/autogen"
logger.info(f"Project root: {PROJECT_ROOT}")

# Load .env from the project root
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, '.env'))

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment!")

os.environ["OPENAI_API_KEY"] = openai_api_key

# Update paths to use project root
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "update_manifest", "program_config.json")
DEFAULT_SCHEMA_PATH = os.path.join(PROJECT_ROOT, "manifest_schema.json")  # Schema is in root
DEFAULT_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "manifest.json")  # Shared manifest
TRACE_PATH = os.path.join(PROJECT_ROOT, "update_manifest_trace.json")

logger.info(f"Config path: {DEFAULT_CONFIG_PATH}")
logger.info(f"Schema path: {DEFAULT_SCHEMA_PATH}")
logger.info(f"Manifest path: {DEFAULT_MANIFEST_PATH}")
logger.info(f"Trace path: {TRACE_PATH}")

# Create directories if they don't exist
os.makedirs(os.path.dirname(DEFAULT_CONFIG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(DEFAULT_MANIFEST_PATH), exist_ok=True)

# Minimal manifest schema for creation if missing (no maxLength for summary)
MINIMAL_MANIFEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["version", "files", "metadata"],
    "properties": {
        "version": {"type": "string"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["filename", "path", "description", "status", "metadata", "sha256", "modified_date", "file_type", "encoding", "size", "dependencies", "category", "read_order"],
                "properties": {
                    "filename": {"type": "string"},
                    "path": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {"type": "string"},
                    "metadata": {
                        "type": "object",
                        "required": ["summary", "keywords", "topics", "entities"],
                        "properties": {
                            "summary": {"type": "string"},  # No maxLength
                            "keywords": {"type": "array", "items": {"type": "string"}},
                            "topics": {"type": "array", "items": {"type": "string"}},
                            "entities": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                    "modified_date": {"type": "string"},
                    "file_type": {"type": "string"},
                    "encoding": {"type": "string"},
                    "size": {"type": "integer"},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    "category": {"type": "string"},
                    "read_order": {"type": "integer"}
                }
            }
        },
        "metadata": {
            "type": "object",
            "properties": {
                "statistics": {"type": "object"},
                "topics": {"type": "object"},
                "entities": {"type": "object"}
            }
        }
    }
}

# Action trace for logging all operations
action_trace: list = []

def log_action(action_type: str, details: dict, agent_name: str = "DirectLLM") -> None:
    """Log an action to the trace with timestamp and agent information."""
    action_trace.append({
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "action_type": action_type,
        "agent": agent_name,
        **details
    })

class TracedGroupChat(GroupChat):
    """A GroupChat that automatically traces all interactions."""
    
    def __init__(self, *args, trace_path: str, **kwargs):
        """Initialize the traced group chat.
        
        Args:
            trace_path: Path where to save the trace file
            *args: Arguments to pass to GroupChat
            **kwargs: Keyword arguments to pass to GroupChat
        """
        super().__init__(*args, **kwargs)
        self.trace_path = trace_path
        self.action_trace: List[Dict[str, Any]] = []
        
    def _log_action(self, action_type: str, details: Dict[str, Any], agent_name: str) -> None:
        """Log an action to the trace with timestamp and agent information."""
        self.action_trace.append({
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "action_type": action_type,
            "agent": agent_name,
            **details
        })
        
    def _save_trace(self) -> None:
        """Save the current trace to the trace file."""
        trace = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "stats": {
                "total_actions": len(self.action_trace),
                "agents": list(set(action["agent"] for action in self.action_trace))
            },
            "actions": self.action_trace
        }
        with open(self.trace_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
            
    def reset(self) -> None:
        """Reset the group chat and save the current trace."""
        self._save_trace()
        super().reset()
        
    def select_speaker(self, *args, **kwargs):
        """Override select_speaker to trace speaker selection."""
        speaker = super().select_speaker(*args, **kwargs)
        if speaker:
            self._log_action(
                "speaker_selected",
                {"selected_speaker": speaker.name},
                "GroupChat"
            )
        return speaker
        
    def send(self, message: str, sender: Optional[str] = None, **kwargs):
        """Override send to trace message sending."""
        self._log_action(
            "message_sent",
            {
                "sender": sender,
                "message_length": len(message),
                **kwargs
            },
            sender or "Unknown"
        )
        return super().send(message, sender, **kwargs)
        
    def receive(self, message: str, sender: Optional[str] = None, **kwargs):
        """Override receive to trace message receiving."""
        self._log_action(
            "message_received",
            {
                "sender": sender,
                "message_length": len(message),
                **kwargs
            },
            self.name
        )
        return super().receive(message, sender, **kwargs)

def compute_sha256(file_path: str, agent_name: str = "FileProcessor") -> str:
    """Compute SHA256 hash of a file."""
    logger.info(f"Computing SHA256 for file: {file_path}")
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read the file in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def llm_generate_file_summary(file_path: str, agent_name: str = "SummaryGenerator") -> Dict[str, Any]:
    logger.info(f"Generating summary for file: {file_path}")
    try:
        content = read_file_content(file_path)
        logger.info(f"Successfully read content from {file_path}")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise

    # PROMPT EM PORTUGUÊS
    prompt = (
        f"Por favor, analise este arquivo de texto e forneça:\n"
        f"1. Uma breve descrição do que este arquivo contém\n"
        f"2. Um resumo detalhado de seu conteúdo\n"
        f"\nConteúdo do arquivo:\n{content}\n\n"
        f"Formate sua resposta assim:\n"
        f"DESCRIÇÃO: (descrição em uma linha)\n"
        f"RESUMO: (resumo detalhado)"
    )
    
    logger.info("Making OpenAI API call for file summary")
    try:
        logger.info(f"Making OpenAI API call for file summary with model: {agent_name}")
        response = openai.chat.completions.create(
            model=agent_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,  # Increased for better summaries
            temperature=0.3,
        )
        
        logger.info("Successfully received OpenAI API response")
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise

    # Parse the response in a simple format (Portuguese markers)
    response_text = response.choices[0].message.content.strip()
    try:
        # Simple parsing based on DESCRIÇÃO: and RESUMO: markers
        description = ""
        summary = ""
        
        for line in response_text.split('\n'):
            if line.startswith('DESCRIÇÃO:'):
                description = line.replace('DESCRIÇÃO:', '').strip()
            elif line.startswith('RESUMO:'):
                summary = line.replace('RESUMO:', '').strip()
        
        if not description or not summary:
            raise ValueError("Falha ao extrair descrição ou resumo da resposta")
            
        logger.info(f"Successfully parsed summary for {file_path}")
        
    except Exception as e:
        logger.error(f"Error parsing summary for {file_path}: {e}")
        # Provide a basic fallback
        description = f"Arquivo de texto: {os.path.basename(file_path)}"
        summary = "Falha ao gerar resumo. Conteúdo original preservado."

    return {
        "description": description,
        "metadata": {"summary": summary}
    }

def build_manifest_with_llm(text_dir: str, agent_name: str = "ManifestBuilder") -> Dict[str, Any]:
    logger.info(f"Starting manifest build for directory: {text_dir}")
    logger.info(f"Using agent: {agent_name}")
    
    if not os.path.exists(text_dir):
        logger.error(f"Directory does not exist: {text_dir}")
        raise FileNotFoundError(f"Directory not found: {text_dir}")
        
    files = [os.path.join(text_dir, f) for f in os.listdir(text_dir) if os.path.isfile(os.path.join(text_dir, f))]
    # Filter out .DS_Store and other hidden files
    files = [f for f in files if not os.path.basename(f).startswith('.')]
    
    logger.info(f"Found {len(files)} files to process")
    
    manifest = {
        "version": "1.0.0",
        "files": [],
        "metadata": {
            "statistics": {
                "total_files": len(files),
                "total_size": sum(os.path.getsize(f) for f in files),
                "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            "topics": {},
            "entities": {},
        },
    }
    
    for file_path in files:
        logger.info(f"Processing file: {file_path}")
        try:
            summary_data = llm_generate_file_summary(file_path, agent_name)
            logger.info(f"Successfully generated summary for {file_path}")
            
            file_entry = {
                "filename": os.path.basename(file_path),
                "path": file_path,
                "description": summary_data["description"],
                "status": "okay",
                "metadata": {
                    "summary": summary_data["metadata"]["summary"],
                    "keywords": ["ai", "text", "analysis"],
                    "topics": ["artificial intelligence"],
                    "entities": ["llm", "agent"]
                },
                "sha256": compute_sha256(file_path, agent_name),
                "modified_date": datetime.datetime.fromtimestamp(
                    os.path.getmtime(file_path), 
                    tz=datetime.timezone.utc
                ).isoformat(),
                "file_type": "text",
                "encoding": "utf-8",
                "size": os.path.getsize(file_path),
                "dependencies": [],
                "category": "input",
                "read_order": 0,
            }
            manifest["files"].append(file_entry)
            
            logger.info(f"Added {file_path} to manifest")
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            continue
    
    logger.info(f"Manifest build completed. Total files processed: {len(manifest['files'])}")
    
    return manifest

async def main(max_round: int = 10) -> None:
    logger.info("Starting manifest update workflow (direct LLM per-file summary mode)")
    text_dir = os.path.join(PROJECT_ROOT, "text")
    logger.info(f"Text directory: {text_dir}")
    
    if not os.path.exists(text_dir):
        logger.error(f"Text directory does not exist: {text_dir}")
        os.makedirs(text_dir, exist_ok=True)
        logger.info(f"Created text directory: {text_dir}")
    
    logger.info("Building manifest")
    try:
        manifest = build_manifest_with_llm(text_dir)
        logger.info("Successfully built manifest")
        logger.info(f"Manifest contains {len(manifest['files'])} files")
    except Exception as e:
        logger.error(f"Failed to build manifest: {e}")
        raise
        
    logger.info(f"Saving manifest to {DEFAULT_MANIFEST_PATH}")
    try:
        save_json_file(manifest, DEFAULT_MANIFEST_PATH)
        logger.info(f"Successfully saved manifest to {DEFAULT_MANIFEST_PATH}")
        
        # Save trace with all actions and statistics
        trace = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "stats": {
                "total_files": len(manifest["files"]),
                "total_size": manifest["metadata"]["statistics"]["total_size"],
                "success_rate": f"{len(manifest['files'])}/{len([f for f in os.listdir(text_dir) if os.path.isfile(os.path.join(text_dir, f)) and not f.startswith('.')])}",
                "processed_files": [
                    {
                        "filename": entry["filename"],
                        "size": entry["size"],
                        "sha256": entry["sha256"]
                    }
                    for entry in manifest["files"]
                ]
            },
            "actions": action_trace  # Include all logged actions
        }
        save_json_file(trace, TRACE_PATH)
        logger.info(f"Saved trace to {TRACE_PATH}")
        
    except Exception as e:
        logger.error(f"Failed to save manifest: {e}")
        raise
        
    # Validate the updated manifest
    logger.info("Starting manifest validation")
    try:
        logger.info(f"Loading schema from {DEFAULT_SCHEMA_PATH}")
        if not os.path.exists(DEFAULT_SCHEMA_PATH):
            logger.warning(f"Schema file {DEFAULT_SCHEMA_PATH} does not exist. Creating minimal schema.")
            save_json_file(MINIMAL_MANIFEST_SCHEMA, DEFAULT_SCHEMA_PATH)
            logger.info(f"Created minimal schema at {DEFAULT_SCHEMA_PATH}")
        schema = load_json_file(DEFAULT_SCHEMA_PATH)
        logger.info("Successfully loaded schema")
        
        logger.info("Validating manifest against schema")
        validate_manifest(manifest, schema)
        logger.info(f"Manifest validation successful: {DEFAULT_MANIFEST_PATH}")
    except JsonSchemaValidationError as e:
        logger.error(f"Manifest validation failed: {e}")
    except FileNotFoundError as e:
        logger.error(f"File not found during validation: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main(max_round=10))  
