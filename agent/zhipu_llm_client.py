"""
Zhipu AI LLM Client for Graphiti.

This module provides a robust Zhipu AI client using the zhipuai SDK
for proper structured output support with GLM models.

Key features:
1. Smart response normalization - automatically converts various LLM response formats
2. Schema simplification - converts complex dynamic schemas to GLM-friendly format
3. Multi-level fallback mechanism - multiple strategies when parsing fails
4. Response post-processing - pre-processes responses before validation
"""

import asyncio
import json
import logging
import re
import typing
from typing import Any, Dict, Optional, Type, get_type_hints, get_origin, get_args

from pydantic import BaseModel, ValidationError
from zhipuai import ZhipuAI

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize, DEFAULT_MAX_TOKENS
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.prompts.models import Message

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'glm-4.7'

# Multilingual extraction instruction (same as OpenAI client)
MULTILINGUAL_EXTRACTION_RESPONSES = """
Respond in the same language as the input text when extracting entities and relationships.
"""


class ResponseNormalizer:
    """
    Normalize LLM responses to match expected Pydantic model format.
    
    This class handles the various inconsistencies in GLM's JSON output:
    1. Markdown code blocks removal
    2. List-to-dict conversion when model expects dict
    3. Nested field extraction
    4. Type coercion
    """
    
    @staticmethod
    def clean_json_content(content: str) -> str:
        """
        Clean JSON content from markdown code blocks and other formatting.
        
        Args:
            content: Raw LLM response content
            
        Returns:
            Cleaned JSON string
        """
        if not content:
            return content
            
        content = content.strip()
        
        # Remove markdown code blocks
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
            
        if content.endswith('```'):
            content = content[:-3]
            
        content = content.strip()
        
        # Remove any leading/trailing explanatory text
        # Find the first { or [ and last } or ]
        json_start = -1
        json_end = -1
        
        for i, char in enumerate(content):
            if char in '{[':
                json_start = i
                break
                
        for i in range(len(content) - 1, -1, -1):
            if content[i] in '}]':
                json_end = i + 1
                break
                
        if json_start != -1 and json_end != -1:
            content = content[json_start:json_end]
            
        return content
    
    @staticmethod
    def normalize_response(
        raw_response: Any,
        response_model: Type[BaseModel]
    ) -> Dict[str, Any]:
        """
        Normalize the raw LLM response to match the expected model format.
        
        Args:
            raw_response: The parsed JSON from LLM (could be dict, list, etc.)
            response_model: The expected Pydantic model type
            
        Returns:
            Normalized dictionary matching the model's structure
        """
        if raw_response is None:
            return {}
            
        # Get expected field names from the model
        model_fields = set(response_model.model_fields.keys())
        
        # Case 1: Response is already a dict
        if isinstance(raw_response, dict):
            # Check if it's wrapped in a schema-like structure
            # e.g., {'title': 'ModelName', 'properties': {...}}
            if 'properties' in raw_response and 'title' in raw_response:
                # Extract the actual data from properties
                raw_response = raw_response.get('properties', raw_response)
                
            # Check if response has extra nesting
            # e.g., {'data': {'field1': 'value1', ...}}
            if len(raw_response) == 1:
                only_key = list(raw_response.keys())[0]
                inner_value = raw_response[only_key]
                if isinstance(inner_value, dict):
                    inner_fields = set(inner_value.keys())
                    # If inner dict has more matching fields, use it
                    if len(inner_fields & model_fields) > len(set(raw_response.keys()) & model_fields):
                        raw_response = inner_value
                        
            return raw_response
            
        # Case 2: Response is a list when dict expected
        if isinstance(raw_response, list):
            # If list has only one element and it's a dict, unwrap it
            if len(raw_response) == 1 and isinstance(raw_response[0], dict):
                return ResponseNormalizer.normalize_response(raw_response[0], response_model)
                
            # If list contains schema-like objects, extract the relevant one
            for item in raw_response:
                if isinstance(item, dict):
                    # Check if any item matches our expected fields
                    if set(item.keys()) & model_fields:
                        return ResponseNormalizer.normalize_response(item, response_model)
                        
            # Last resort: try to map list to expected structure
            # Create a dict with expected fields populated from list items
            result = {}
            for item in raw_response:
                if isinstance(item, dict):
                    result.update(item)
            if result:
                return result
                
            logger.warning(f"Could not normalize list response: {raw_response}")
            return {}
            
        # Case 3: Other types - wrap in appropriate structure
        logger.warning(f"Unexpected response type: {type(raw_response)}")
        return {}
    
    @staticmethod
    def extract_fields_from_wrapper(
        response: Dict[str, Any],
        response_model: Type[BaseModel]
    ) -> Dict[str, Any]:
        """
        Extract fields when response is wrapped in an extra layer.
        
        Common GLM pattern: returns schema description instead of data.
        e.g., [{"title": "ModelName", "properties": {"field": {"default": "value"}}}]
        
        Args:
            response: Normalized dict response
            response_model: Expected Pydantic model
            
        Returns:
            Dict with extracted field values
        """
        model_fields = response_model.model_fields
        result = {}
        
        for field_name, field_info in model_fields.items():
            if field_name in response:
                value = response[field_name]
                
                # Handle nested property objects
                if isinstance(value, dict) and 'default' in value:
                    result[field_name] = value['default']
                elif isinstance(value, dict) and 'value' in value:
                    result[field_name] = value['value']
                else:
                    result[field_name] = value
            elif field_info.default is not None:
                result[field_name] = field_info.default
                
        return result if result else response


class SchemaSimplifier:
    """
    Simplify complex Pydantic schemas for better GLM understanding.
    
    GLM models sometimes struggle with complex nested schemas.
    This class creates simplified schema representations.
    """
    
    @staticmethod
    def simplify_schema(response_model: Type[BaseModel]) -> str:
        """
        Create a simplified, GLM-friendly schema description.
        
        Args:
            response_model: The Pydantic model to simplify
            
        Returns:
            Simplified schema string
        """
        schema = response_model.model_json_schema()
        
        # Create a simpler representation
        simplified_fields = []
        properties = schema.get('properties', {})
        required = set(schema.get('required', []))
        
        for field_name, field_info in properties.items():
            field_type = field_info.get('type', 'string')
            description = field_info.get('description', '')
            is_required = field_name in required
            
            # Handle complex types
            if field_type == 'array':
                items = field_info.get('items', {})
                item_type = items.get('type', 'object')
                field_type = f'array of {item_type}'
                
            req_marker = '(required)' if is_required else '(optional)'
            simplified_fields.append(f'  "{field_name}": {field_type} {req_marker} - {description}')
            
        return '{\n' + ',\n'.join(simplified_fields) + '\n}'
    
    @staticmethod
    def create_example(response_model: Type[BaseModel]) -> str:
        """
        Create an example JSON output for the model.
        
        Args:
            response_model: The Pydantic model
            
        Returns:
            Example JSON string
        """
        example = {}
        for field_name, field_info in response_model.model_fields.items():
            annotation = field_info.annotation
            
            # Get the origin type for generic types
            origin = get_origin(annotation)
            
            if origin is list:
                # For list types, create an empty list or example list
                args = get_args(annotation)
                if args and hasattr(args[0], 'model_fields'):
                    # Nested model
                    example[field_name] = []
                else:
                    example[field_name] = []
            elif annotation == str:
                example[field_name] = f"example_{field_name}"
            elif annotation == int:
                example[field_name] = 0
            elif annotation == float:
                example[field_name] = 0.0
            elif annotation == bool:
                example[field_name] = True
            elif hasattr(annotation, 'model_fields'):
                # Nested model
                example[field_name] = {}
            else:
                example[field_name] = None
                
        return json.dumps(example, indent=2)


class ZhipuAIClient(LLMClient):
    """
    ZhipuAIClient is a client class for interacting with Zhipu AI's GLM models.
    
    This client uses the native zhipuai SDK with json_object mode for 
    reliable structured output, with robust response normalization and
    multi-level fallback mechanisms.
    """
    
    # Maximum number of retries for failed requests
    MAX_RETRIES = 2

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the ZhipuAI client.
        
        Args:
            config: LLM configuration with api_key and model
            cache: Whether to use caching for responses
            max_tokens: Maximum tokens in response
        """
        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        self.client = ZhipuAI(api_key=config.api_key)
        self.model = config.model or DEFAULT_MODEL
        self.max_tokens = max_tokens
        
        self.normalizer = ResponseNormalizer()
        self.schema_simplifier = SchemaSimplifier()

    def _build_system_prompt(
        self,
        original_system: str,
        response_model: Type[BaseModel] | None
    ) -> str:
        """
        Build enhanced system prompt with schema instructions.
        
        Args:
            original_system: Original system prompt content
            response_model: Optional Pydantic model for structured output
            
        Returns:
            Enhanced system prompt
        """
        if response_model is None:
            return original_system
            
        # Get simplified schema
        schema_json = response_model.model_json_schema()
        simplified = self.schema_simplifier.simplify_schema(response_model)
        
        # Build enhanced prompt
        enhancement = f"""

CRITICAL OUTPUT REQUIREMENTS:
1. You MUST output ONLY valid JSON, no explanatory text before or after.
2. Do NOT include markdown code blocks (no ``` markers).
3. The JSON MUST be a single object (dictionary), NOT an array/list.
4. Follow this exact structure:

{simplified}

Full JSON Schema for reference:
{json.dumps(schema_json, indent=2)}

Remember: Output ONLY the JSON object, nothing else.
"""
        return f"{original_system}{enhancement}"

    def _validate_and_normalize(
        self,
        content: str,
        response_model: Type[BaseModel]
    ) -> Dict[str, Any]:
        """
        Validate and normalize response content against the model.
        
        This method implements multiple fallback strategies:
        1. Direct JSON parse and validate
        2. Clean and retry
        3. Normalize structure and validate
        4. Extract from wrapper and validate
        
        Args:
            content: Raw response content string
            response_model: Expected Pydantic model
            
        Returns:
            Validated and normalized dictionary
            
        Raises:
            ValidationError: If all validation attempts fail
        """
        # Strategy 1: Direct parse and validate
        try:
            parsed = json.loads(content)
            validated = response_model.model_validate(parsed)
            return validated.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Direct validation failed: {e}")
            
        # Strategy 2: Clean JSON and retry
        cleaned = self.normalizer.clean_json_content(content)
        try:
            parsed = json.loads(cleaned)
            validated = response_model.model_validate(parsed)
            return validated.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Cleaned validation failed: {e}")
            
        # Strategy 3: Normalize structure  
        try:
            parsed = json.loads(cleaned)
            normalized = self.normalizer.normalize_response(parsed, response_model)
            validated = response_model.model_validate(normalized)
            return validated.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Normalized validation failed: {e}")
            
        # Strategy 4: Extract from wrapper
        try:
            parsed = json.loads(cleaned)
            normalized = self.normalizer.normalize_response(parsed, response_model)
            extracted = self.normalizer.extract_fields_from_wrapper(normalized, response_model)
            validated = response_model.model_validate(extracted)
            return validated.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Extracted validation failed: {e}")
            
        # All strategies failed - raise with context
        raise ValidationError.from_exception_data(
            "All validation strategies failed",
            [
                {
                    'type': 'value_error',
                    'loc': (),
                    'msg': f'Could not parse response: {content[:500]}...',
                    'input': content,
                }
            ]
        )

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from the Zhipu AI GLM model.
        
        Args:
            messages: List of messages to send
            response_model: Optional Pydantic model for structured output
            max_tokens: Maximum tokens in response
            model_size: Model size hint (not used)
            
        Returns:
            Dict containing the parsed response
            
        Raises:
            RateLimitError: If API rate limit is exceeded
            Exception: If response parsing fails
        """
        try:
            zhipu_messages: list[dict] = []
            
            # Build system prompt with schema instructions if needed
            system_prompt = ''
            if messages and messages[0].role == 'system':
                system_prompt = self._build_system_prompt(
                    messages[0].content,
                    response_model
                )
                messages = messages[1:]
            elif response_model is not None:
                # No system message but need structured output
                system_prompt = self._build_system_prompt('', response_model)
            
            if system_prompt:
                zhipu_messages.append({'role': 'system', 'content': system_prompt})
            
            # Add remaining messages
            for m in messages:
                zhipu_messages.append({
                    'role': m.role,
                    'content': self._clean_input(m.content)
                })
            
            # Build request parameters
            request_params = {
                'model': self.model,
                'messages': zhipu_messages,
                'max_tokens': max_tokens or self.max_tokens,
                'temperature': self.temperature,
            }
            
            # Use json_object mode for structured output
            if response_model is not None:
                request_params['response_format'] = {'type': 'json_object'}
            
            # Run sync API in thread pool (zhipuai SDK is synchronous)
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                **request_params
            )
            
            content = response.choices[0].message.content
            
            # Parse structured response with normalization
            if response_model is not None:
                if not content:
                    raise ValueError('No response content from Zhipu AI')
                
                return self._validate_and_normalize(content, response_model)
            
            return {'content': content}
            
        except Exception as e:
            error_str = str(e).lower()
            # Check for rate limit errors (code 1302 or HTTP 429)
            if '429' in error_str or '1302' in error_str or 'rate limit' in error_str:
                raise RateLimitError from e
            logger.error(f'Error in generating LLM response: {e}')
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response with retry logic for failed requests.
        
        This method wraps _generate_response with retry logic, similar to
        the OpenAI client implementation. It will retry on parsing errors
        but not on rate limit errors.
        
        Args:
            messages: List of messages to send
            response_model: Optional Pydantic model for structured output
            max_tokens: Maximum tokens in response
            model_size: Model size hint (not used)
            
        Returns:
            Dict containing the parsed response
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        retry_count = 0
        last_error = None

        # Add multilingual extraction instructions
        if messages and messages[0].role == 'system':
            messages[0].content += MULTILINGUAL_EXTRACTION_RESPONSES

        while retry_count <= self.MAX_RETRIES:
            try:
                response = await self._generate_response(
                    messages, response_model, max_tokens, model_size
                )
                return response
            except RateLimitError:
                # Rate limit errors should not trigger retries - let caller handle
                raise
            except Exception as e:
                last_error = e

                # Don't retry if we've hit the max retries
                if retry_count >= self.MAX_RETRIES:
                    logger.error(f'Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}')
                    raise

                retry_count += 1

                # Construct a detailed error message for the LLM
                error_context = (
                    f'Your previous response was invalid. '
                    f'Error: {str(e)}. '
                    f'Please try again. Remember to output ONLY a valid JSON object (not an array), '
                    f'with no markdown formatting, no explanatory text.'
                )

                error_message = Message(role='user', content=error_context)
                messages.append(error_message)
                logger.warning(
                    f'Retrying after error (attempt {retry_count}/{self.MAX_RETRIES}): {e}'
                )

        # If we somehow get here, raise the last error
        raise last_error or Exception('Max retries exceeded with no specific error')


# Alias for backward compatibility
ZhipuAILLMClient = ZhipuAIClient
