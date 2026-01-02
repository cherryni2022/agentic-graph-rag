"""
Test script to verify GLM JSON parsing with UniversalLLMClient.

This script confirms that the _extract_json and _adapt_json_structure
methods work correctly for GLM's markdown-wrapped JSON responses.
"""

import json
from pydantic import BaseModel
from typing import List


class ExtractedEntity(BaseModel):
    """Model for a single extracted entity."""
    name: str
    entity_type_id: int


class ExtractedEntities(BaseModel):
    """Model that Graphiti expects."""
    extracted_entities: List[ExtractedEntity]


def test_extract_json():
    """Test _extract_json method."""
    
    # Import the client
    from agent.universal_llm_client import UniversalLLMClient
    from graphiti_core.llm_client.config import LLMConfig
    
    # Create a mock config (won't be used for this test)
    config = LLMConfig(
        api_key="test",
        model="test"
    )
    
    client = UniversalLLMClient(config)
    
    # Test case 1: GLM response with markdown code blocks
    glm_response = '''```json
[
  {
    "name": "OpenAI",
    "entity_type_id": 0
  },
  {
    "name": "Microsoft",
    "entity_type_id": 0
  }
]
```'''
    
    extracted = client._extract_json(glm_response)
    print("Test 1 - Extract JSON from markdown block:")
    print(f"  Input: {glm_response[:50]}...")
    print(f"  Output: {extracted[:50]}...")
    
    # Verify it's valid JSON
    parsed = json.loads(extracted)
    print(f"  Parsed successfully: {len(parsed)} entities")
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    print("  ✅ PASSED\n")
    
    # Test case 2: Adapt array to object structure
    print("Test 2 - Adapt array to object structure:")
    adapted = client._adapt_json_structure(parsed, ExtractedEntities)
    print(f"  Input: array with {len(parsed)} items")
    print(f"  Output: {adapted}")
    assert isinstance(adapted, dict)
    assert "extracted_entities" in adapted
    assert len(adapted["extracted_entities"]) == 2
    print("  ✅ PASSED\n")
    
    # Test case 3: Validate with Pydantic
    print("Test 3 - Pydantic validation:")
    validated = ExtractedEntities.model_validate(adapted)
    print(f"  Validated model: {validated}")
    assert len(validated.extracted_entities) == 2
    assert validated.extracted_entities[0].name == "OpenAI"
    print("  ✅ PASSED\n")
    
    # Test case 4: Object format (should pass through unchanged)
    print("Test 4 - Object format pass-through:")
    object_data = {"extracted_entities": parsed}
    adapted_obj = client._adapt_json_structure(object_data, ExtractedEntities)
    assert adapted_obj == object_data
    print("  ✅ PASSED\n")
    
    print("=" * 60)
    print("All tests passed! UniversalLLMClient should work with GLM.")
    print("=" * 60)


if __name__ == "__main__":
    test_extract_json()
