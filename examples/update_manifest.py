"""
Update Manifest Example (Multi-Agent Version)

This example demonstrates a multi-agent system that:
1. Generates a manifest for project files using agent classes
2. Validates the manifest against a schema
3. Saves the validated manifest
4. Collects a trace of all agent interactions

Uses autogen's GroupChat for orchestration and real agent implementations.
"""

import os
import hashlib
import datetime
from dotenv import load_dotenv
from typing import Dict, Any
from autogen_extensions.openai_utils import get_openai_client
from autogen import AssistantAgent
from autogen_extensions.auto_tracing_group_chat import AutoTracingGroupChat
from autogen_extensions.log_utils import get_logger

from examples.common import (
    load_json_file,
    save_json_file,
    JsonSchemaValidationError,
    read_file_content,
    validate_manifest,
)

# Set up logging before any logger.info calls
get_logger(__name__).info("Starting logging setup")

# Project root directory
PROJECT_ROOT = "/Users/lorenzo/Sync/Source/AI/autogen"
get_logger(__name__).info(f"Project root: {PROJECT_ROOT}")

# Load .env from the project root
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

# --- OpenAI Client Setup ---
openai_client = get_openai_client()

# Update paths to use project root
DEFAULT_CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "config", "update_manifest", "program_config.json"
)
DEFAULT_SCHEMA_PATH = os.path.join(
    PROJECT_ROOT, "manifest_schema.json"
)  # Schema is in root
DEFAULT_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "manifest.json")  # Shared manifest
TRACE_PATH = os.path.join(PROJECT_ROOT, "update_manifest_trace.json")

get_logger(__name__).info(f"Config path: {DEFAULT_CONFIG_PATH}")
get_logger(__name__).info(f"Schema path: {DEFAULT_SCHEMA_PATH}")
get_logger(__name__).info(f"Manifest path: {DEFAULT_MANIFEST_PATH}")
get_logger(__name__).info(f"Trace path: {TRACE_PATH}")

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
                "required": [
                    "filename",
                    "path",
                    "description",
                    "status",
                    "metadata",
                    "sha256",
                    "modified_date",
                    "file_type",
                    "encoding",
                    "size",
                    "dependencies",
                    "category",
                    "read_order",
                ],
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
                            "entities": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                    "modified_date": {"type": "string"},
                    "file_type": {"type": "string"},
                    "encoding": {"type": "string"},
                    "size": {"type": "integer"},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    "category": {"type": "string"},
                    "read_order": {"type": "integer"},
                },
            },
        },
        "metadata": {
            "type": "object",
            "properties": {
                "statistics": {"type": "object"},
                "topics": {"type": "object"},
                "entities": {"type": "object"},
            },
        },
    },
}

# --- Agent Classes ---
llm_config = {
    "api_key": openai_client.api_key,
    "model": "gpt-4",
}


class FileReaderAgent(AssistantAgent):
    """
    Agent responsible for reading file content.
    """

    def __init__(self, name: str, llm_config, functions, system_message: str):
        super().__init__(
            name=name,
            llm_config=llm_config,
            functions=functions,
            system_message=system_message,
        )

    def read(self, file_path: str) -> str:
        """
        Read the content of a file and log the action.
        """
        content = read_file_content(file_path)
        return content


class SummarizerAgent(AssistantAgent):
    """
    Agent responsible for summarizing file content using an LLM.
    """

    def __init__(self, name: str, llm_config, functions, system_message: str):
        super().__init__(
            name=name,
            llm_config=llm_config,
            functions=functions,
            system_message=system_message,
        )

    def summarize(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Summarize the content of a file using OpenAI's API and log the action.
        """
        get_logger(__name__).debug(
            f"SummarizerAgent: Preparing to summarize file: {file_path}"
        )
        # Estimate token count (1 token ≈ 4 chars for rough estimate)
        max_tokens = 8192 - 1000  # leave room for completion

        def num_tokens(text: str) -> int:
            return len(text) // 4

        if num_tokens(content) > max_tokens:
            get_logger(__name__).warning(
                f"File {file_path} is too large for LLM context window, skipping."
            )
            return {
                "description": f"Arquivo de texto: {os.path.basename(file_path)} (excedeu limite de contexto LLM)",
                "summary": "Arquivo muito grande para resumir automaticamente.",
            }
        prompt = (
            f"Por favor, analise este arquivo de texto e forneça:\n"
            f"1. Uma breve descrição do que este arquivo contém\n"
            f"2. Um resumo detalhado de seu conteúdo\n"
            f"\nConteúdo do arquivo:\n{content}\n\n"
            f"Formate sua resposta assim:\n"
            f"DESCRIÇÃO: (descrição em uma linha)\n"
            f"RESUMO: (resumo detalhado)"
        )
        get_logger(__name__).debug(
            f"SummarizerAgent: Starting OpenAI API call for file: {file_path}"
        )
        response = openai_client.chat.completions.create(
            model=llm_config["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3,
        )
        get_logger(__name__).debug(
            f"SummarizerAgent: Finished OpenAI API call for file: {file_path}"
        )
        response_text = response.choices[0].message.content.strip()
        description = ""
        summary = ""
        for line in response_text.split("\n"):
            if line.startswith("DESCRIÇÃO:"):
                description = line.replace("DESCRIÇÃO:", "").strip()
            elif line.startswith("RESUMO:"):
                summary = line.replace("RESUMO:", "").strip()
        if not description or not summary:
            description = f"Arquivo de texto: {os.path.basename(file_path)}"
            summary = "Falha ao gerar resumo. Conteúdo original preservado."
        get_logger(__name__).debug(
            f"SummarizerAgent: Completed summarization for file: {file_path}"
        )
        return {"description": description, "summary": summary}


class ValidatorAgent(AssistantAgent):
    """
    Agent responsible for validating the manifest against a schema.
    """

    def __init__(self, name: str, llm_config, functions, system_message: str):
        super().__init__(
            name=name,
            llm_config=llm_config,
            functions=functions,
            system_message=system_message,
        )

    def validate(self, manifest: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate the manifest using the provided schema and log the result.
        """
        get_logger(__name__).debug(
            f"ValidatorAgent self.name: {self.name}"
        )  # Debug print
        try:
            validate_manifest(manifest, schema)
            return True
        except JsonSchemaValidationError:
            return False
        except Exception:
            raise


# --- Main Workflow ---
def compute_sha256(file_path: str) -> str:
    """
    Compute the SHA256 hash of a file.
    """
    get_logger(__name__).info(f"Computing SHA256 for file: {file_path}")
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def build_manifest_with_agents(text_dir: str, model: str = "gpt-4") -> Dict[str, Any]:
    get_logger(__name__).info(f"Starting manifest build for directory: {text_dir}")
    if not os.path.exists(text_dir):
        get_logger(__name__).error(f"Directory does not exist: {text_dir}")
        raise FileNotFoundError(f"Directory not found: {text_dir}")
    files = [
        os.path.join(text_dir, f)
        for f in os.listdir(text_dir)
        if os.path.isfile(os.path.join(text_dir, f))
    ]
    files = [f for f in files if not os.path.basename(f).startswith(".")]
    get_logger(__name__).info(f"Found {len(files)} files to process")
    manifest = {
        "version": "1.0.0",
        "files": [],
        "metadata": {
            "statistics": {
                "total_files": len(files),
                "total_size": sum(os.path.getsize(f) for f in files),
                "last_updated": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            },
            "topics": {},
            "entities": {},
        },
    }
    file_reader = FileReaderAgent(
        name="FileReaderAgent",
        llm_config=llm_config,
        functions=[],
        system_message="Reads file content for manifest generation.",
    )
    summarizer = SummarizerAgent(
        name="SummarizerAgent",
        llm_config=llm_config,
        functions=[],
        system_message="Summarizes file content using LLM.",
    )
    validator = ValidatorAgent(
        name="ValidatorAgent",
        llm_config=llm_config,
        functions=[],
        system_message="Validates the manifest against the schema.",
    )
    agents = [file_reader, summarizer, validator]
    group_description = (
        "Update Manifest Multi-Agent Workflow: "
        "This group coordinates FileReaderAgent, SummarizerAgent, and ValidatorAgent "
        "to build, summarize, and validate a manifest for project files."
    )
    with AutoTracingGroupChat(
        agents=agents, trace_file=TRACE_PATH, description=group_description
    ) as group:
        for file_path in files:
            get_logger(__name__).info(f"Processing file: {file_path}")
            # FileReaderAgent reads the file
            content = file_reader.read(file_path)
            group.agent_action("file_read", {"file": file_path}, file_reader.name)
            # SummarizerAgent summarizes the file
            summary_data = summarizer.summarize(file_path, content)
            group.agent_action(
                "file_summarized",
                {"file": file_path, "summary": summary_data["summary"]},
                summarizer.name,
            )
            # Assemble file entry for manifest
            file_entry = {
                "filename": os.path.basename(file_path),
                "path": file_path,
                "description": summary_data["description"],
                "status": "okay",
                "metadata": {
                    "summary": summary_data["summary"],
                    "keywords": ["ai", "text", "analysis"],
                    "topics": ["artificial intelligence"],
                    "entities": ["llm", "agent"],
                },
                "sha256": compute_sha256(file_path),
                "modified_date": datetime.datetime.fromtimestamp(
                    os.path.getmtime(file_path), tz=datetime.timezone.utc
                ).isoformat(),
                "file_type": "text",
                "encoding": "utf-8",
                "size": os.path.getsize(file_path),
                "dependencies": [],
                "category": "input",
                "read_order": 0,
            }
            manifest["files"].append(file_entry)
            group.agent_action("file_added_to_manifest", {"file": file_path}, "System")
            get_logger(__name__).info(f"Added {file_path} to manifest")
        # Validate manifest at the end
        valid = validator.validate(manifest, MINIMAL_MANIFEST_SCHEMA)
        group.agent_action("manifest_validated", {"result": valid}, validator.name)
        get_logger(__name__).info(
            f"Manifest build completed. Total files processed: {len(manifest['files'])}"
        )
    return manifest


async def main(max_round: int = 10) -> None:
    """
    Main entry point for the manifest update workflow.
    Orchestrates the multi-agent process: build, save, and validate the manifest.
    """
    get_logger(__name__).info("Starting manifest update workflow (multi-agent mode)")
    text_dir = os.path.join(PROJECT_ROOT, "text")
    get_logger(__name__).info(f"Text directory: {text_dir}")
    if not os.path.exists(text_dir):
        get_logger(__name__).error(f"Text directory does not exist: {text_dir}")
        os.makedirs(text_dir, exist_ok=True)
        get_logger(__name__).info(f"Created text directory: {text_dir}")
    get_logger(__name__).info("Building manifest with agents")
    try:
        manifest = build_manifest_with_agents(text_dir)
        get_logger(__name__).info("Successfully built manifest")
        get_logger(__name__).info(f"Manifest contains {len(manifest['files'])} files")
    except Exception as e:
        get_logger(__name__).error(f"Failed to build manifest: {e}")
        raise
    get_logger(__name__).info(f"Saving manifest to {DEFAULT_MANIFEST_PATH}")
    try:
        save_json_file(manifest, DEFAULT_MANIFEST_PATH)
        get_logger(__name__).info(
            f"Successfully saved manifest to {DEFAULT_MANIFEST_PATH}"
        )
    except Exception as e:
        get_logger(__name__).error(f"Failed to save manifest: {e}")
        raise
    get_logger(__name__).info("Starting manifest validation")
    try:
        get_logger(__name__).info(f"Loading schema from {DEFAULT_SCHEMA_PATH}")
        if not os.path.exists(DEFAULT_SCHEMA_PATH):
            get_logger(__name__).warning(
                f"Schema file {DEFAULT_SCHEMA_PATH} does not exist. Creating minimal schema."
            )
            save_json_file(MINIMAL_MANIFEST_SCHEMA, DEFAULT_SCHEMA_PATH)
            get_logger(__name__).info(
                f"Created minimal schema at {DEFAULT_SCHEMA_PATH}"
            )
        schema = load_json_file(DEFAULT_SCHEMA_PATH)
        get_logger(__name__).info("Successfully loaded schema")
        get_logger(__name__).info("Validating manifest against schema")
        validator = ValidatorAgent(
            "ValidatorAgent",
            llm_config,
            [],
            "Validates the manifest against the schema.",
        )
        valid = validator.validate(manifest, schema)
        if valid:
            get_logger(__name__).info(
                f"Manifest validation successful: {DEFAULT_MANIFEST_PATH}"
            )
        else:
            get_logger(__name__).error(
                f"Manifest validation failed: {DEFAULT_MANIFEST_PATH}"
            )
    except Exception as e:
        get_logger(__name__).error(f"Unexpected error during validation: {e}")
        raise


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(max_round=10))
