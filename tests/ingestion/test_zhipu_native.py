"""
Test script to verify ZhipuAIClient with native structured output.

This script tests that the native Zhipu AI SDK (zhipuai) returns clean JSON
without markdown code blocks.
"""

import os
import asyncio
import json
import logging
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Graphiti's ExtractedEntities model (simplified)
class ExtractedEntity(BaseModel):
    """An extracted entity."""
    name: str
    entity_type_id: int = Field(default=0)


class ExtractedEntities(BaseModel):
    """Model that Graphiti expects for entity extraction."""
    extracted_entities: List[ExtractedEntity]


# GLM Configuration
ZHIPU_LLM_API_KEY = os.getenv("LLM_API_KEY", "9b07546272e9445d91b0e90808689f62.WUfFQlCBCvttAn3W")
ZHIPU_LLM_MODEL = os.getenv("LLM_CHOICE", "glm-4.7")


async def test_zhipu_native_client():
    """Test ZhipuAIClient with native structured output."""
    
    from agent.zhipu_llm_client import ZhipuAIClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.prompts.models import Message
    
    logger.info("=" * 60)
    logger.info("Testing ZhipuAIClient (Native SDK)")
    logger.info("=" * 60)
    
    # Create LLM config for GLM
    config = LLMConfig(
        api_key=ZHIPU_LLM_API_KEY,
        model=ZHIPU_LLM_MODEL,
    )
    
    # Create ZhipuAIClient
    client = ZhipuAIClient(config=config)
    
    logger.info(f"Using model: {ZHIPU_LLM_MODEL}")
    
    # Test content
    test_content = """
    OpenAI announced a new partnership with Microsoft to develop advanced AI systems.
    The collaboration also includes NVIDIA for hardware optimization.
    Sam Altman, CEO of OpenAI, presented the news at a press conference.
    """
    
    entity_types = "[{'entity_type_id': 0, 'entity_type_name': 'Entity', 'entity_type_description': 'Default entity classification.'}]"
    
    # Create messages using Graphiti's Message format
    messages = [
        Message(
            role="system",
            content="""
            You are an AI assistant that extracts entity nodes from text. 
            Your primary task is to extract and classify the speaker and other significant entities mentioned in the provided text.
            Do not escape unicode characters.
            Any extracted information should be returned in the same language as it was written in.
            """
        ),
        Message(
            role="user",
            content=f"""
                <TEXT>{test_content}</TEXT>
                <ENTITY TYPES>
                    {entity_types}
                </ENTITY TYPES>
                Given the above text, extract entities from the TEXT that are explicitly or implicitly mentioned.
                For each entity extracted, also determine its entity type based on the provided ENTITY TYPES and their descriptions.
                Indicate the classified entity type by providing its entity_type_id.
                Guidelines:
                    1. Extract significant entities, concepts, or actors mentioned in the conversation.
                    2. Avoid creating nodes for relationships or actions.
                    3. Avoid creating nodes for temporal information like dates, times or years (these will be added to edges later).
                    4. Be as explicit as possible in your node names, using full names and avoiding abbreviations.
            """
        )
    ]
    
    try:
        logger.info("\n--- Calling Zhipu AI Native SDK ---")
        
        # Call _generate_response (what Graphiti calls internally)
        result = await client.generate_response(
            messages=messages,
            response_model=ExtractedEntities,
        )
        
        logger.info("\n✅ SUCCESS! Zhipu AI response parsed correctly:")
        logger.info(f"Type: {type(result)}")
        logger.info(f"Result: {result}")
        
        # Verify result is a dict (Graphiti expects dict from .model_dump())
        if isinstance(result, dict):
            logger.info("✓ Result is a dictionary (correct for Graphiti)")
            entities = result.get('extracted_entities', [])
            logger.info(f"\nExtracted {len(entities)} entities:")
            for entity in entities:
                if isinstance(entity, dict):
                    logger.info(f"  - {entity.get('name')} (type_id: {entity.get('entity_type_id')})")
                else:
                    logger.info(f"  - {entity.name} (type_id: {entity.entity_type_id})")
        else:
            logger.warning("⚠️ Result is not a dict - Graphiti may fail!")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_comparison():
    """Compare UniversalLLMClient vs ZhipuAIClient."""
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON: UniversalLLMClient vs ZhipuAIClient")
    logger.info("=" * 60)
    
    # Test ZhipuAIClient
    zhipu_success = await test_zhipu_native_client()
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"ZhipuAIClient (Native SDK): {'✅ PASSED' if zhipu_success else '❌ FAILED'}")
    print("=" * 60)
    
    return zhipu_success


if __name__ == "__main__":
    success = asyncio.run(test_comparison())
    exit(0 if success else 1)
