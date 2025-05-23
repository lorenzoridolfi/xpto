"""
JSON Validator Tool for Agent Output Validation

This module defines a tool that agents can use to validate their JSON output
against their defined schemas.
"""

import json
from typing import Dict, Any, Tuple
from jsonschema import validate, ValidationError


def validate_json(data: dict, schema: dict) -> bool:
    """
    Validate a dictionary against a JSON schema. Returns True if valid, raises Exception if not.
    """
    validate(instance=data, schema=schema)
    return True


def validate_json_output(output: str) -> Tuple[bool, str]:
    """
    Validates JSON output against the agent's schema.

    Args:
        output (str): The JSON output to validate

    Returns:
        Tuple[bool, str]: (is_valid, message)
            - is_valid: True if the output is valid JSON and matches the schema
            - message: Success message or error details
    """
    try:
        # First, try to parse the JSON
        parsed_output = json.loads(output)

        # Get the agent's schema
        with open("agent_schemas.json", "r", encoding="utf-8") as f:
            schemas = json.load(f)["schemas"]

        # Find the appropriate schema based on the output structure
        schema = None
        if "files_used" in parsed_output and "content" in parsed_output:
            schema = schemas.get("writer")
        elif (
            "verification_status" in parsed_output
            and "verification_results" in parsed_output
        ):
            schema = schemas.get("information_verifier")
        elif (
            "quality_status" in parsed_output and "quality_assessment" in parsed_output
        ):
            schema = schemas.get("text_quality_expert")

        if not schema:
            return False, "No matching schema found for the output structure"

        # Validate against the schema
        validate(instance=parsed_output, schema=schema)
        return True, "Output is valid JSON and matches the schema"

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format: {str(e)}"
    except ValidationError as e:
        return False, f"Schema validation failed: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


# Tool definition for agent configuration
JSON_VALIDATOR_TOOL = {
    "name": "validate_json_output",
    "description": "Validates JSON output against the agent's schema",
    "parameters": {
        "type": "object",
        "properties": {
            "output": {"type": "string", "description": "The JSON output to validate"}
        },
        "required": ["output"],
    },
}


def get_tool_for_agent(agent_name: str) -> Dict[str, Any]:
    """
    Get the appropriate tool configuration for an agent.

    Args:
        agent_name (str): Name of the agent

    Returns:
        Dict[str, Any]: Tool configuration or empty dict if no tool needed
    """
    if agent_name in ["WriterAgent", "InformationVerifierAgent", "TextQualityAgent"]:
        return JSON_VALIDATOR_TOOL
    return {}
