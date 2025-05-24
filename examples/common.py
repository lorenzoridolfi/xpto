import os
import json
import jsonschema
import hashlib
from typing import Any, Dict, Tuple, List
import datetime
from openai import OpenAI

client = OpenAI()
from pydantic import BaseModel


class JsonSchemaValidationError(Exception):
    """Raised when JSON schema validation fails."""

    pass


class FileSummaryMetadata(BaseModel):
    summary: str


class FileSummaryResponse(BaseModel):
    description: str
    metadata: FileSummaryMetadata


def load_json_file(path: str) -> Any:
    """Load a JSON file and return its contents as a Python object."""
    print(f"DEBUG: load_json_file called with path={path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data: Any, path: str) -> None:
    """Save a Python object as a JSON file."""
    print(f"DEBUG: save_json_file called with path={path}, data type={type(data)})")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def validate_json_file_with_schema(
    data_path: str,
    schema_path: str,
) -> Dict[str, Any]:
    """
    Load and validate a JSON file against a JSON schema.

    Args:
        data_path (str): Path to the JSON data file.
        schema_path (str): Path to the JSON schema file.

    Returns:
        dict: Loaded and validated JSON data.

    Raises:
        FileNotFoundError: If the data or schema file does not exist.
        json.JSONDecodeError: If the data or schema is not valid JSON.
        JsonSchemaValidationError: If the data does not conform to the schema.
    """
    print(
        f"DEBUG: validate_json_file_with_schema called with data_path={data_path}, schema_path={schema_path}"
    )
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        raise JsonSchemaValidationError(f"JSON schema validation failed: {e.message}")
    return data


def compute_sha256(file_path: str) -> str:
    """Compute the SHA256 hash of a file."""
    m = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            m.update(chunk)
    return m.hexdigest()


def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """Get basic metadata for a file, including a simple summary."""
    # Ensure file_path is a string for compatibility
    if not isinstance(file_path, str):
        file_path = str(file_path)
    stat = os.stat(file_path)
    # Generate a summary: first 3 lines or first 200 chars
    summary = ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [next(f) for _ in range(3)]
            summary = " ".join(line.strip() for line in lines)
    except (UnicodeDecodeError, StopIteration):
        # Fallback: read as bytes or file is too short
        try:
            with open(file_path, "rb") as f:
                content = f.read(200)
                summary = content.decode("utf-8", errors="replace")
        except Exception:
            summary = "(unreadable file)"
    return {
        "filename": os.path.basename(file_path),
        "path": file_path,
        "description": summary,
        "status": "okay",
        "metadata": {
            "summary": summary,
            "keywords": ["test", "example", "file"],
            "topics": ["testing"],
            "entities": ["entity1", "entity2"],
        },
        "sha256": compute_sha256(file_path),
        "modified_date": "2024-01-01T00:00:00Z",
        "file_type": "text",
        "encoding": "utf-8",
        "size": stat.st_size,
        "dependencies": [],
        "category": "input",
        "read_order": 0,
    }


def generate_manifest(directory: str) -> Tuple[Dict[str, Any], List[str]]:
    """Generate a manifest for all files in a directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist")

    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    manifest = {
        "version": "1.0.0",
        "files": [get_file_metadata(f) for f in files],
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
    return manifest, files


def validate_manifest(manifest: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Validate a manifest dict against a schema dict."""
    try:
        jsonschema.validate(instance=manifest, schema=schema)
    except jsonschema.ValidationError as e:
        raise JsonSchemaValidationError(f"Manifest validation failed: {e.message}")


def update_and_save_manifest(
    directory: str, manifest_path: str, schema_path: str
) -> None:
    """Generate, validate, and save a manifest for a directory."""
    manifest, _ = generate_manifest(directory)
    schema = load_json_file(schema_path)
    validate_manifest(manifest, schema)
    save_json_file(manifest, manifest_path)


def load_manifest_schema(schema_path: str) -> Dict[str, Any]:
    """Load a manifest schema from a file."""
    return load_json_file(schema_path)


def summarize_file_with_llm(file_path: str) -> str:
    """Summarize the content of a file in less than 200 words using OpenAI LLM."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read(2000)  # Limit to first 2000 chars for prompt
    prompt = (
        "Summarize the following file content in less than 200 words for a technical manifest:\n\n"
        f"{content}\n\nSummary:"
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def read_file_content(file_path: str) -> str:
    """Read and return the content of a file as a string."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def parse_file_summary_response(json_str: str) -> FileSummaryResponse:
    """Parse and validate the LLM output for a file summary using the FileSummaryResponse model."""
    return FileSummaryResponse.model_validate_json(json_str)
