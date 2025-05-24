#!/usr/bin/env python3
"""
Autogen Core Models Information Extractor
This program extracts what's actually available in autogen_core.models module.
Note: The actual model implementations and their capabilities are in autogen_ext packages.
"""

import json
import inspect
from datetime import datetime


def extract_autogen_models_info():
    """
    Extract information from autogen_core.models module.
    Important: This module contains model families and interfaces,
    NOT the actual supported models and their capabilities.
    """

    result = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "important_note": "autogen_core.models contains model families and interfaces, NOT actual model implementations",
            "actual_models_location": "Model implementations with capabilities are in autogen_ext.models.* packages",
        },
        "model_families": {},
        "capability_structure": {},
        "message_types": [],
        "available_in_core": {},
    }

    try:
        # Import the core models module
        import autogen_core.models as models

        # 1. Extract Model Families (these are just constants, not actual models)
        print("Extracting Model Families...")
        if hasattr(models, "ModelFamily"):
            ModelFamily = models.ModelFamily
            families = {}

            for attr_name in dir(ModelFamily):
                if not attr_name.startswith("_") and attr_name.isupper():
                    value = getattr(ModelFamily, attr_name)
                    if isinstance(value, str):
                        families[attr_name] = value

            result["model_families"] = {
                "description": "Model family constants defined in autogen_core",
                "note": "These are just string constants for categorization, not actual model implementations",
                "families": families,
            }

        # 2. Extract ModelInfo structure (the capability fields)
        print("Extracting ModelInfo structure...")
        if hasattr(models, "ModelInfo"):
            ModelInfo = models.ModelInfo
            capability_fields = {}

            if hasattr(ModelInfo, "__annotations__"):
                for field, field_type in ModelInfo.__annotations__.items():
                    capability_fields[field] = {
                        "type": str(field_type),
                        "description": f"Model capability: {field}",
                    }

            result["capability_structure"] = {
                "description": "Structure for defining model capabilities",
                "usage": "Model clients use this to define their capabilities",
                "fields": capability_fields,
            }

        # 3. Extract message types
        print("Extracting message types...")
        message_types = []
        for name, obj in inspect.getmembers(models):
            if inspect.isclass(obj) and name.endswith("Message"):
                message_types.append(
                    {
                        "name": name,
                        "docstring": inspect.getdoc(obj) or "No documentation",
                    }
                )

        result["message_types"] = message_types

        # 4. Extract what's actually available
        print("Extracting available components...")
        available_items = {"classes": [], "functions": [], "constants": []}

        for name, obj in inspect.getmembers(models):
            if not name.startswith("_"):
                if inspect.isclass(obj):
                    available_items["classes"].append(name)
                elif inspect.isfunction(obj):
                    available_items["functions"].append(name)
                elif isinstance(obj, (str, int, float)):
                    available_items["constants"].append(name)

        result["available_in_core"] = available_items

        # 5. Add clarification about actual models
        result["clarification"] = {
            "what_this_module_contains": [
                "Model family constants (e.g., 'gpt-4o', 'claude-3-opus')",
                "Base classes and interfaces (ChatCompletionClient)",
                "Message type definitions (UserMessage, SystemMessage, etc.)",
                "Capability structure definitions (ModelInfo, ModelCapabilities)",
            ],
            "what_this_module_does_not_contain": [
                "Actual list of supported models",
                "Specific model capabilities (e.g., 'gpt-4o supports vision')",
                "Model client implementations",
                "Model-specific configuration",
            ],
            "where_to_find_actual_models": {
                "OpenAI models": "autogen_ext.models.openai",
                "Anthropic models": "autogen_ext.models.anthropic",
                "Azure models": "autogen_ext.models.azure",
                "Ollama models": "autogen_ext.models.ollama",
                "Note": "These packages contain the actual model implementations with capabilities",
            },
        }

        # 6. Try to show how capabilities would be defined
        result["capability_example"] = {
            "description": "Example of how model capabilities are defined in actual implementations",
            "example": {
                "model": "gpt-4o (this would be in autogen_ext.models.openai)",
                "model_info": {
                    "vision": True,
                    "function_calling": True,
                    "json_output": True,
                    "family": "gpt-4o",
                    "structured_output": True,
                },
            },
        }

    except ImportError as e:
        result["error"] = f"Failed to import autogen_core.models: {str(e)}"
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"

    return result


def main():
    print("=" * 80)
    print(
        "IMPORTANT: autogen_core.models does NOT contain the actual supported models!"
    )
    print("It only contains model family constants and interface definitions.")
    print("The actual model implementations are in autogen_ext packages.")
    print("=" * 80)
    print()

    print("Extracting information from autogen_core.models...")
    data = extract_autogen_models_info()

    # Save to JSON
    filename = "autogen_models.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {filename}")

    # Print summary
    print("\nSUMMARY:")
    print("-" * 40)
    if "model_families" in data and "families" in data["model_families"]:
        print(f"Model Families Found: {len(data['model_families']['families'])}")
        print("These are just string constants, NOT actual model implementations!")

    print("\nTo get actual supported models and their capabilities, you would need to:")
    print("1. Import autogen_ext.models.* packages")
    print("2. Instantiate the model clients")
    print("3. Check their model_info properties")
    print("\nExample:")
    print("from autogen_ext.models.openai import OpenAIChatCompletionClient")
    print("client = OpenAIChatCompletionClient(model='gpt-4o')")
    print("print(client.model_info)  # This would show actual capabilities")


if __name__ == "__main__":
    main()
