from typing import Type, Optional, Any, Dict, List
from pydantic import BaseModel, ValidationError
from autogen_extensions.llm_base_agent import LLMBaseAgent
import logging

try:
    from pydantic import create_model_from_schema  # Pydantic v2+
except ImportError:
    create_model_from_schema = None

logger = logging.getLogger("structured_llm_base_agent")

class StructuredLLMBaseAgent(LLMBaseAgent):
    """
    Base agent for LLMs that return structured (Pydantic) output.
    Handles LLM call and output validation. If output_model is not provided, but output_schema is,
    a Pydantic model will be generated from the schema (Pydantic v2+).
    """
    output_model: Optional[Type[BaseModel]] = None  # Subclasses can set this
    output_schema: Optional[Dict[str, Any]] = None  # Subclasses can set this

    def __init__(
        self,
        output_model: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        llm_call: Optional[Any] = None,
        *args,
        **kwargs,
    ):
        super().__init__(llm_call=llm_call, *args, **kwargs)
        if output_model is not None:
            self.output_model = output_model
        if self.output_model is None and (output_schema is not None or self.output_schema is not None):
            schema = output_schema or self.output_schema
            if create_model_from_schema is None:
                raise ImportError("pydantic.create_model_from_schema is required (Pydantic v2+)")
            self.output_model = create_model_from_schema(schema, model_name="DynamicModel")
        if self.output_model is None:
            raise ValueError("output_model or output_schema must be provided for StructuredLLMBaseAgent")

    async def call_structured_llm(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
        **llm_kwargs,
    ) -> BaseModel:
        """
        Calls the LLM and parses the response as a Pydantic model.
        Raises RuntimeError if validation fails.
        """
        if messages is None:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        if response_format is None:
            response_format = {"type": "json_object"}
        try:
            response = await self._call_llm(
                messages=messages,
                model=model,
                response_format=response_format,
                **llm_kwargs,
            )
            content = response.choices[0].message.content
            logger.debug(f"LLM raw response: {content}")
            obj = self.output_model.model_validate_json(content)
            logger.info(f"Validated structured output: {obj}")
            return obj
        except ValidationError as e:
            logger.error(f"Pydantic validation error: {e}")
            raise RuntimeError(f"Failed to validate structured LLM output: {e}")
        except Exception as e:
            logger.error(f"LLM call or parsing error: {e}")
            raise 