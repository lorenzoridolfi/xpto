import jsonschema

ValidationError = jsonschema.ValidationError


def validate_json(data: dict, schema: dict) -> bool:
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True
    except jsonschema.ValidationError as e:
        raise ValueError(f"JSON validation error: {e.message}")
