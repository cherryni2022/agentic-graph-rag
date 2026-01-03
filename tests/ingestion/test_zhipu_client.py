"""
Test script for ZhipuAIClient with dynamic Pydantic models.

This script tests the exact scenario that was failing:
extract_attributes_from_node uses pydantic.create_model to create
dynamic models, which GLM was returning as lists instead of dicts.
"""

import asyncio
import json
import logging
import pydantic
from pydantic import BaseModel, Field
from uuid import uuid4
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the ZhipuAIClient
from agent.zhipu_llm_client import ZhipuAIClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.prompts.models import Message


async def test_dynamic_model():
    """
    Test the exact scenario from extract_attributes_from_node.
    
    This tests:
    1. Dynamic Pydantic model creation with create_model
    2. LLM response parsing with the dynamic model
    3. Response normalization when GLM returns lists
    """
    logger.info("=" * 80)
    logger.info("Testing Dynamic Pydantic Model (extract_attributes_from_node scenario)")
    logger.info("=" * 80)
    
    # Create dynamic model exactly as graphiti does
    attributes_definitions = {
        'summary': (
            str,
            Field(
                description='Summary containing the important information about the entity. Under 250 words',
            ),
        ),
        # Add some extra fields like custom entity types might have
        'industry': (
            str,
            Field(description='The industry the entity operates in'),
        ),
        'founded_year': (
            int | None,
            Field(description='The year the entity was founded', default=None),
        ),
    }
    
    unique_model_name = f'EntityAttributes_{uuid4().hex}'
    entity_attributes_model = pydantic.create_model(unique_model_name, **attributes_definitions)
    
    logger.info(f"Created dynamic model: {unique_model_name}")
    logger.info(f"Model schema: {json.dumps(entity_attributes_model.model_json_schema(), indent=2)}")
    
    # Create the LLM client
    config = LLMConfig(
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_CHOICE", "glm-4.7"),
        base_url=os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
    )
    
    client = ZhipuAIClient(config=config)
    
    # Build context similar to graphiti's extract_attributes
    node_context = {
        'name': 'OpenAI',
        'summary': 'A leading AI research company',
        'entity_types': ['Organization', 'Company'],
        'attributes': {},
    }
    
    episode_content = """
    OpenAI is an artificial intelligence research laboratory founded in December 2015. 
    The company operates in the technology and AI industry. It was founded by Sam Altman, 
    Elon Musk, and others with the goal of developing safe and beneficial AI.
    OpenAI is known for creating ChatGPT and GPT-4, and is now valued at over $80 billion.
    """
    
    # Build messages similar to graphiti's extract_attributes prompt
    messages = [
        Message(
            role='system',
            content='You are a helpful assistant that extracts entity properties from the provided text.',
        ),
        Message(
            role='user',
            content=f"""

        <MESSAGES>
        {json.dumps(episode_content, indent=2)}
        </MESSAGES>

        Given the above MESSAGES and the following ENTITY, update any of its attributes based on the information provided
        in MESSAGES. Use the provided attribute descriptions to better understand how each attribute should be determined.

        Guidelines:
        1. Do not hallucinate entity property values if they cannot be found in the current context.
        2. Only use the provided MESSAGES and ENTITY to set attribute values.
        3. The summary attribute represents a summary of the ENTITY, and should be updated with new information about the Entity from the MESSAGES. 
            Summaries must be no longer than 250 words.
        
        <ENTITY>
        {json.dumps(node_context, indent=2)}
        </ENTITY>
        """,
        ),
    ]
    
    try:
        logger.info("Calling LLM with dynamic model...")
        response = await client.generate_response(
            messages=messages,
            response_model=entity_attributes_model,
            model_size=ModelSize.small,
        )
        
        logger.info("SUCCESS! Response received:")
        logger.info(json.dumps(response, indent=2, ensure_ascii=False))
        
        # Verify we can extract the expected fields
        assert 'summary' in response, "Missing 'summary' field"
        logger.info(f"Summary: {response.get('summary')}")
        logger.info(f"Industry: {response.get('industry')}")
        logger.info(f"Founded Year: {response.get('founded_year')}")
        
        return True
        
    except Exception as e:
        logger.error(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_static_model():
    """
    Test with a static Pydantic model (ExtractedEntities equivalent).
    """
    logger.info("=" * 80)
    logger.info("Testing Static Pydantic Model (extract_nodes scenario)")
    logger.info("=" * 80)
    
    class ExtractedEntity(BaseModel):
        name: str = Field(..., description='Name of the extracted entity')
        entity_type_id: int = Field(description='ID of the classified entity type.')
    
    class ExtractedEntities(BaseModel):
        extracted_entities: list[ExtractedEntity] = Field(..., description='List of extracted entities')
    
    # Create the LLM client
    config = LLMConfig(
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_CHOICE", "glm-4.7"),
        base_url=os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
    )
    
    client = ZhipuAIClient(config=config)
    
    test_content = """
    Google announced a new AI model called Gemini in December 2023.
    The model was developed by DeepMind and Google Research teams.
    Sundar Pichai, CEO of Google, presented the model at a press conference.
    """
    
    entity_types = [
        {'entity_type_id': 0, 'entity_type_name': 'Entity', 'entity_type_description': 'Default entity'},
        {'entity_type_id': 1, 'entity_type_name': 'Organization', 'entity_type_description': 'A company or organization'},
        {'entity_type_id': 2, 'entity_type_name': 'Person', 'entity_type_description': 'A person'},
        {'entity_type_id': 3, 'entity_type_name': 'Product', 'entity_type_description': 'A product or technology'},
    ]
    
    messages = [
        Message(
            role='system',
            content='You are an AI assistant that extracts entity nodes from text.',
        ),
        Message(
            role='user',
            content=f"""
<TEXT>
{test_content}
</TEXT>
<ENTITY TYPES>
{json.dumps(entity_types, indent=2)}
</ENTITY TYPES>

Given the above text, extract entities from the TEXT that are explicitly or implicitly mentioned.
For each entity extracted, also determine its entity type based on the provided ENTITY TYPES.
""",
        ),
    ]
    
    try:
        logger.info("Calling LLM with static model...")
        response = await client.generate_response(
            messages=messages,
            response_model=ExtractedEntities,
            model_size=ModelSize.small,
        )
        
        logger.info("SUCCESS! Response received:")
        logger.info(json.dumps(response, indent=2, ensure_ascii=False))
        
        assert 'extracted_entities' in response, "Missing 'extracted_entities' field"
        entities = response['extracted_entities']
        logger.info(f"Extracted {len(entities)} entities")
        
        for entity in entities:
            logger.info(f"  - {entity.get('name')} (type_id: {entity.get('entity_type_id')})")
        
        return True
        
    except Exception as e:
        logger.error(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_simple_model():
    """
    Test with a very simple model to verify basic functionality.
    """
    logger.info("=" * 80)
    logger.info("Testing Simple Pydantic Model (basic sanity check)")
    logger.info("=" * 80)
    
    class SimpleOutput(BaseModel):
        answer: str = Field(..., description='The answer to the question')
        confidence: float = Field(..., description='Confidence score between 0 and 1')
    
    config = LLMConfig(
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_CHOICE", "glm-4.7"),
        base_url=os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
    )
    
    client = ZhipuAIClient(config=config)
    
    messages = [
        Message(role='system', content='You are a helpful assistant.'),
        Message(role='user', content='What is the capital of France?'),
    ]
    
    try:
        logger.info("Calling LLM with simple model...")
        response = await client.generate_response(
            messages=messages,
            response_model=SimpleOutput,
        )
        
        logger.info("SUCCESS! Response received:")
        logger.info(json.dumps(response, indent=2, ensure_ascii=False))
        
        assert 'answer' in response, "Missing 'answer' field"
        assert 'confidence' in response, "Missing 'confidence' field"
        
        return True
        
    except Exception as e:
        logger.error(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    results = {}
    
    # Test 1: Simple model
    results['simple_model'] = await test_simple_model()
    print()
    
    # Test 2: Static model (like ExtractedEntities)
    results['static_model'] = await test_static_model()
    print()
    
    # Test 3: Dynamic model (like EntityAttributes - the failing scenario!)
    results['dynamic_model'] = await test_dynamic_model()
    print()
    
    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed! The ZhipuAIClient should now work with Graphiti.")
    else:
        logger.info("\n⚠️ Some tests failed. Please review the logs above.")
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
