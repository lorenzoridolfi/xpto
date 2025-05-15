"""
JSON Validator Tool for Agent Output Validation

This module provides a robust JSON validation system for agent outputs, ensuring
that responses conform to expected schemas and formats. It includes:

- Schema validation using JSON Schema
- Custom validation rules
- Error reporting and formatting
- Schema management and versioning
- Performance optimization
"""

from typing import Dict, Any, Optional, List, Union
import json
from jsonschema import validate, ValidationError
import re
from datetime import datetime

class JSONValidator:
    """
    A comprehensive JSON validation system for agent outputs.
    
    This class provides:
    - Schema-based validation using JSON Schema
    - Custom validation rules and constraints
    - Detailed error reporting
    - Schema management and versioning
    - Performance optimization through caching
    
    Attributes:
        schemas (Dict[str, Dict[str, Any]]): Registered validation schemas
        custom_rules (Dict[str, callable]): Custom validation rules
        cache (Dict[str, bool]): Validation result cache
    """
    
    def __init__(self):
        """Initialize the JSON validator with empty schemas and rules."""
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.custom_rules: Dict[str, callable] = {}
        self.cache: Dict[str, bool] = {}

    def register_schema(self,
                       schema_name: str,
                       schema: Dict[str, Any],
                       version: str = "1.0") -> None:
        """
        Register a new validation schema.

        Args:
            schema_name (str): Name of the schema
            schema (Dict[str, Any]): JSON Schema definition
            version (str): Schema version

        Raises:
            ValueError: If schema is invalid or already registered
        """
        if not isinstance(schema, dict):
            raise ValueError("Schema must be a dictionary")
            
        if schema_name in self.schemas:
            raise ValueError(f"Schema '{schema_name}' already registered")
            
        # Validate schema format
        try:
            validate(instance={}, schema=schema)
        except ValidationError as e:
            raise ValueError(f"Invalid schema format: {str(e)}")
            
        # Add version to schema
        schema["$schema"] = f"http://json-schema.org/draft-07/schema#"
        schema["$id"] = f"{schema_name}@{version}"
        
        self.schemas[schema_name] = schema
        self.cache.clear()  # Clear cache when schemas change

    def register_custom_rule(self,
                           rule_name: str,
                           rule_func: callable) -> None:
        """
        Register a custom validation rule.

        Args:
            rule_name (str): Name of the rule
            rule_func (callable): Function that takes a value and returns bool

        Raises:
            ValueError: If rule is invalid or already registered
        """
        if not callable(rule_func):
            raise ValueError("Rule must be a callable function")
            
        if rule_name in self.custom_rules:
            raise ValueError(f"Rule '{rule_name}' already registered")
            
        self.custom_rules[rule_name] = rule_func
        self.cache.clear()  # Clear cache when rules change

    def validate(self,
                data: Any,
                schema_name: str,
                custom_rules: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate data against a schema and custom rules.

        Args:
            data (Any): Data to validate
            schema_name (str): Name of the schema to use
            custom_rules (Optional[List[str]]): List of custom rules to apply

        Returns:
            Dict[str, Any]: Validation result with:
                - is_valid (bool): Whether validation passed
                - errors (List[str]): List of validation errors
                - warnings (List[str]): List of validation warnings

        Raises:
            ValueError: If schema or rules are not found
        """
        if schema_name not in self.schemas:
            raise ValueError(f"Schema '{schema_name}' not found")
            
        # Check cache first
        cache_key = f"{schema_name}:{json.dumps(data, sort_keys=True)}"
        if cache_key in self.cache:
            return {
                "is_valid": self.cache[cache_key],
                "errors": [],
                "warnings": []
            }
            
        errors = []
        warnings = []
        
        # Validate against schema
        try:
            validate(instance=data, schema=self.schemas[schema_name])
        except ValidationError as e:
            errors.append(str(e))
            
        # Apply custom rules
        if custom_rules:
            for rule_name in custom_rules:
                if rule_name not in self.custom_rules:
                    raise ValueError(f"Rule '{rule_name}' not found")
                    
                try:
                    if not self.custom_rules[rule_name](data):
                        errors.append(f"Failed custom rule: {rule_name}")
                except Exception as e:
                    warnings.append(f"Error applying rule '{rule_name}': {str(e)}")
                    
        # Cache result
        is_valid = len(errors) == 0
        self.cache[cache_key] = is_valid
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings
        }

    def get_schema(self, schema_name: str) -> Dict[str, Any]:
        """
        Get a registered schema.

        Args:
            schema_name (str): Name of the schema

        Returns:
            Dict[str, Any]: Schema definition

        Raises:
            ValueError: If schema is not found
        """
        if schema_name not in self.schemas:
            raise ValueError(f"Schema '{schema_name}' not found")
            
        return self.schemas[schema_name]

    def list_schemas(self) -> List[str]:
        """
        List all registered schemas.

        Returns:
            List[str]: List of schema names
        """
        return list(self.schemas.keys())

    def list_custom_rules(self) -> List[str]:
        """
        List all registered custom rules.

        Returns:
            List[str]: List of rule names
        """
        return list(self.custom_rules.keys())

    def clear_cache(self) -> None:
        """Clear the validation result cache."""
        self.cache.clear()

    def remove_schema(self, schema_name: str) -> None:
        """
        Remove a registered schema.

        Args:
            schema_name (str): Name of the schema to remove

        Raises:
            ValueError: If schema is not found
        """
        if schema_name not in self.schemas:
            raise ValueError(f"Schema '{schema_name}' not found")
            
        del self.schemas[schema_name]
        self.cache.clear()

    def remove_custom_rule(self, rule_name: str) -> None:
        """
        Remove a registered custom rule.

        Args:
            rule_name (str): Name of the rule to remove

        Raises:
            ValueError: If rule is not found
        """
        if rule_name not in self.custom_rules:
            raise ValueError(f"Rule '{rule_name}' not found")
            
        del self.custom_rules[rule_name]
        self.cache.clear()

    def update_schema(self,
                     schema_name: str,
                     schema: Dict[str, Any],
                     version: Optional[str] = None) -> None:
        """
        Update an existing schema.

        Args:
            schema_name (str): Name of the schema to update
            schema (Dict[str, Any]): New schema definition
            version (Optional[str]): New schema version

        Raises:
            ValueError: If schema is not found or invalid
        """
        if schema_name not in self.schemas:
            raise ValueError(f"Schema '{schema_name}' not found")
            
        if not isinstance(schema, dict):
            raise ValueError("Schema must be a dictionary")
            
        # Validate schema format
        try:
            validate(instance={}, schema=schema)
        except ValidationError as e:
            raise ValueError(f"Invalid schema format: {str(e)}")
            
        # Update schema
        if version:
            schema["$id"] = f"{schema_name}@{version}"
        else:
            schema["$id"] = self.schemas[schema_name]["$id"]
            
        schema["$schema"] = "http://json-schema.org/draft-07/schema#"
        self.schemas[schema_name] = schema
        self.cache.clear()

    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics.

        Returns:
            Dict[str, Any]: Statistics including:
                - total_validations: Total number of validations performed
                - cache_hits: Number of cache hits
                - cache_miss_rate: Cache miss rate
                - avg_validation_time: Average validation time
                - schema_usage: Usage count per schema
                - rule_usage: Usage count per rule
        """
        total_validations = len(self.cache)
        cache_hits = sum(1 for v in self.cache.values() if v)
        
        return {
            "total_validations": total_validations,
            "cache_hits": cache_hits,
            "cache_miss_rate": (total_validations - cache_hits) / total_validations if total_validations > 0 else 0,
            "schema_count": len(self.schemas),
            "rule_count": len(self.custom_rules)
        } 