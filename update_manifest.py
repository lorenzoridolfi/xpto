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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MANIFEST_VERSION = "1.0.0"
MANIFEST_BACKUP_DIR = "manifest_backups"
MAX_BACKUPS = 5

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

def load_schema() -> Dict:
    """Load the manifest schema."""
    try:
        return load_json_file("manifest_schema.json")
    except Exception as e:
        logger.error(f"Error loading schema: {str(e)}")
        raise

def validate_manifest(manifest: Dict) -> bool:
    """Validate manifest against schema."""
    try:
        schema = load_schema()
        jsonschema.validate(instance=manifest, schema=schema)
        return True
    except jsonschema.exceptions.ValidationError as e:
        logger.error(f"Manifest validation error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        return False

def create_backup(manifest_path: str) -> None:
    """Create a backup of the manifest file."""
    try:
        # Create backup directory if it doesn't exist
        os.makedirs(MANIFEST_BACKUP_DIR, exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(MANIFEST_BACKUP_DIR, f"manifest_{timestamp}.json")
        
        # Copy the manifest file
        shutil.copy2(manifest_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        
        # Clean up old backups
        cleanup_old_backups()
    except Exception as e:
        logger.error(f"Error creating backup: {str(e)}")
        raise

def cleanup_old_backups() -> None:
    """Remove old backup files keeping only the most recent ones."""
    try:
        backups = sorted([
            os.path.join(MANIFEST_BACKUP_DIR, f) 
            for f in os.listdir(MANIFEST_BACKUP_DIR) 
            if f.startswith("manifest_") and f.endswith(".json")
        ])
        
        # Remove old backups
        while len(backups) > MAX_BACKUPS:
            os.remove(backups.pop(0))
            logger.info(f"Removed old backup: {backups[0]}")
    except Exception as e:
        logger.error(f"Error cleaning up backups: {str(e)}")
        raise

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

def main():
    """Main function to update the manifest."""
    try:
        # Load configuration
        config = load_json_file("update_manifest_config.json")
        
        # Process files and generate manifest
        manifest = process_files(config)
        
        # Validate manifest
        if not validate_manifest(manifest):
            raise ValueError("Generated manifest failed validation")
        
        # Create backup of existing manifest if it exists
        manifest_path = "file_manifest.json"
        if os.path.exists(manifest_path):
            create_backup(manifest_path)
        
        # Save new manifest
        save_json_file(manifest, manifest_path)
        logger.info("Manifest updated successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
