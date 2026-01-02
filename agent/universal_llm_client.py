"""
Universal LLM Client for Graphiti.

This module provides a wrapper around Graphiti's OpenAIClient that supports
multiple LLM providers by handling structured output in a model-agnostic way.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel

from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig
import openai

logger = logging.getLogger(__name__)


class UniversalLLMClient(OpenAIClient):
    """
    Universal LLM client that extends Graphiti's OpenAIClient.
    
    This client handles structured output for both OpenAI-compatible models
    that support beta.chat.completions.parse and those that don't.
    """
    
    def __init__(
        self,
        config: LLMConfig,
        use_structured_output: bool = True,
        fallback_to_manual_parsing: bool = True
    ):
        """
        Initialize universal LLM client.
        
        Args:
            config: LLM configuration
            use_structured_output: Whether to try using structured output API first
            fallback_to_manual_parsing: Whether to fallback to manual JSON parsing
        """
        super().__init__(config)
        self.use_structured_output = use_structured_output
        self.fallback_to_manual_parsing = fallback_to_manual_parsing
        self._structured_output_supported = None  # Cache the capability check
    
    async def _check_structured_output_support(self) -> bool:
        """
        Check if the model supports structured output.
        
        Returns:
            True if structured output is supported
        """
        if self._structured_output_supported is not None:
            return self._structured_output_supported
        
        try:
            # Try a simple structured output call
            test_model = BaseModel
            response = await self.client.beta.chat.completions.parse(
                model=self.config.model,
                messages=[{"role": "user", "content": "test"}],
                response_format=test_model,
                max_tokens=10
            )
            self._structured_output_supported = True
            logger.info(f"Model {self.config.model} supports structured output")
            return True
        except Exception as e:
            logger.warning(f"Model {self.config.model} does not support structured output: {e}")
            self._structured_output_supported = False
            return False
    
    async def _generate_response_with_fallback(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> Any:
        """
        Generate response with automatic fallback to manual parsing.
        
        Args:
            messages: Chat messages
            response_model: Pydantic model for structured output
            **kwargs: Additional arguments
            
        Returns:
            Parsed response
        """
        # If no response model, use regular completion
        if response_model is None:
            return await super()._generate_response(messages, **kwargs)
        
        # Try structured output first if enabled
        if self.use_structured_output:
            try:
                response = await self.client.beta.chat.completions.parse(
                    model=self.config.model,
                    messages=messages,
                    response_format=response_model,
                    **kwargs
                )
                
                parsed = response.choices[0].message.parsed
                if parsed:
                    logger.debug("Successfully used structured output API")
                    return parsed
                    
            except Exception as e:
                logger.warning(f"Structured output failed: {e}")
                if not self.fallback_to_manual_parsing:
                    raise
        
        # Fallback to manual JSON parsing
        logger.info("Using manual JSON parsing fallback")
        return await self._manual_json_parsing(messages, response_model, **kwargs)
    
    async def _manual_json_parsing(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        **kwargs
    ) -> BaseModel:
        """
        Manually parse JSON response from LLM.
        
        This method handles models that return JSON wrapped in markdown code blocks
        and also handles structure mismatches (e.g., array vs object).
        
        Args:
            messages: Chat messages
            response_model: Pydantic model to validate against
            **kwargs: Additional arguments
            
        Returns:
            Validated Pydantic model instance
        """
        # Enhance the system prompt to request JSON output
        enhanced_messages = self._enhance_messages_for_json(messages, response_model)
        
        # Get regular completion
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=enhanced_messages,
            temperature=kwargs.get('temperature', 0.0),
            max_tokens=kwargs.get('max_tokens', 4096)
        )
        
        content = response.choices[0].message.content
        
        # Extract JSON from response
        json_content = self._extract_json(content)
        
        # Parse and validate
        try:
            parsed_data = json.loads(json_content)
            
            # Auto-convert structure if needed
            # If LLM returned an array but model expects an object with a list field
            parsed_data = self._adapt_json_structure(parsed_data, response_model)
            
            validated = response_model.model_validate(parsed_data)
            logger.debug(f"Successfully parsed and validated JSON response")
            return validated
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Raw content: {content}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except Exception as e:
            logger.error(f"Failed to validate response: {e}")
            raise
    
    def _adapt_json_structure(
        self,
        parsed_data: Any,
        response_model: Type[BaseModel]
    ) -> Dict[str, Any]:
        """
        Adapt JSON structure to match the expected Pydantic model.
        
        This handles cases where:
        - LLM returns array [...] but model expects {"field_name": [...]}
        - LLM returns object with different key names
        
        Args:
            parsed_data: Parsed JSON data
            response_model: Expected Pydantic model
            
        Returns:
            Adapted data matching model structure
        """
        # If already matches, return as-is
        if isinstance(parsed_data, dict):
            return parsed_data
        
        # If LLM returned an array, wrap it in the expected structure
        if isinstance(parsed_data, list):
            # Get the model's fields to find the list field name
            model_fields = response_model.model_fields
            
            # Look for a field that expects a list
            for field_name, field_info in model_fields.items():
                # Check if this field expects a list type
                field_type = field_info.annotation
                if hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                    logger.info(f"Auto-wrapping array into '{field_name}' field")
                    return {field_name: parsed_data}
            
            # Common field names for extracted entities
            common_names = ['extracted_entities', 'entities', 'items', 'results', 'data']
            for name in common_names:
                if name in model_fields:
                    logger.info(f"Auto-wrapping array into '{name}' field (common name match)")
                    return {name: parsed_data}
            
            # Last resort: use first field that looks like it could be a list
            if model_fields:
                first_field = list(model_fields.keys())[0]
                logger.warning(f"Auto-wrapping array into first field '{first_field}'")
                return {first_field: parsed_data}
        
        return parsed_data
    
    def _enhance_messages_for_json(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel]
    ) -> List[Dict[str, str]]:
        """
        Enhance messages to request JSON output.
        
        Args:
            messages: Original messages
            response_model: Pydantic model schema
            
        Returns:
            Enhanced messages
        """
        # Get JSON schema from Pydantic model
        schema = response_model.model_json_schema()
        
        # Create enhanced system message
        json_instruction = f"""
CRITICAL INSTRUCTION: You MUST respond with valid JSON that matches this exact schema:

{json.dumps(schema, indent=2)}

Rules:
1. Output ONLY valid JSON, no other text
2. Do not include markdown code blocks (no ```json or ```)
3. Ensure all required fields are present
4. Follow the exact schema structure
5. Use proper JSON formatting (quotes, commas, brackets)
"""
        
        enhanced_messages = messages.copy()
        
        # Add JSON instruction to system message or create new one
        if enhanced_messages and enhanced_messages[0]["role"] == "system":
            enhanced_messages[0]["content"] += "\n\n" + json_instruction
        else:
            enhanced_messages.insert(0, {
                "role": "system",
                "content": json_instruction
            })
        
        return enhanced_messages
    
    def _extract_json(self, content: str) -> str:
        """
        Extract JSON from LLM response, handling markdown code blocks.
        
        Supports both JSON objects ({...}) and arrays ([...]).
        
        Args:
            content: Raw LLM response
            
        Returns:
            Extracted JSON string
        """
        # Remove markdown code blocks
        if "```json" in content:
            parts = content.split("```json")
            if len(parts) > 1:
                content = parts[1].split("```")[0].strip()
        elif "```" in content:
            parts = content.split("```")
            if len(parts) > 1:
                content = parts[1].split("```")[0].strip()
        
        # Trim whitespace
        content = content.strip()
        
        # Detect if it's a JSON array or object
        is_array = content.startswith("[") or (not content.startswith("{") and "[" in content)
        
        if is_array:
            # Handle JSON arrays
            if not content.startswith("[") and "[" in content:
                start = content.index("[")
                content = content[start:]
            
            if not content.endswith("]") and "]" in content:
                end = content.rindex("]") + 1
                content = content[:end]
        else:
            # Handle JSON objects
            if not content.startswith("{") and "{" in content:
                start = content.index("{")
                content = content[start:]
            
            if not content.endswith("}") and "}" in content:
                end = content.rindex("}") + 1
                content = content[:end]
        
        return content
    
    async def _generate_response(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> Any:
        """
        Override _generate_response to handle models that wrap JSON in markdown code blocks.
        
        This is the method that Graphiti internally calls, so we must override this
        (not generate_response) to intercept structured output parsing.
        
        Args:
            messages: Chat messages
            response_model: Optional Pydantic model for structured output
            **kwargs: Additional arguments
            
        Returns:
            Generated response
        """
        return await self._generate_response_with_fallback(
            messages,
            response_model,
            **kwargs
        )
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> Any:
        """
        Public method for generating response - delegates to _generate_response.
        
        Args:
            messages: Chat messages
            response_model: Optional Pydantic model for structured output
            **kwargs: Additional arguments
            
        Returns:
            Generated response
        """
        return await self._generate_response(
            messages,
            response_model,
            **kwargs
        )


def create_universal_llm_client(config: LLMConfig) -> UniversalLLMClient:
    """
    Factory function to create a universal LLM client.
    
    Args:
        config: LLM configuration
        
    Returns:
        Configured universal LLM client
    """
    return UniversalLLMClient(
        config=config,
        use_structured_output=True,
        fallback_to_manual_parsing=True
    )
