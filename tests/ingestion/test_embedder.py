
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from openai import RateLimitError, APIError
from ingestion.embedder import EmbeddingGenerator

@pytest.mark.asyncio
async def test_generate_embedding_success():
    """Test successful embedding generation."""
    embedder = EmbeddingGenerator()
    
    with patch('ingestion.embedder.embedding_client') as mock_client:
        # Mock response structure
        mock_response = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = [0.1, 0.2, 0.3]
        mock_response.data = [mock_data]
        
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        
        # Action
        result = await embedder.generate_embedding("test text")
        
        # Assert
        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

@pytest.mark.asyncio
async def test_generate_embedding_max_retries_zero():
    """Test that max_retries=0 raises RuntimeError."""
    embedder = EmbeddingGenerator(max_retries=0)
    
    # The loop won't run, so it should hit the final raise
    with pytest.raises(RuntimeError, match="Failed to generate embedding after 0 retries"):
        await embedder.generate_embedding("test text")

@pytest.mark.asyncio
async def test_generate_embedding_all_retries_fail_exception():
    """Test that exhausting retries raises the original exception."""
    embedder = EmbeddingGenerator(max_retries=2, retry_delay=0.01)
    
    with patch('ingestion.embedder.embedding_client') as mock_client:
        # Mock always failing
        mock_client.embeddings.create = AsyncMock(side_effect=Exception("Test error"))
        
        # Action & Assert
        with pytest.raises(Exception, match="Test error"):
            await embedder.generate_embedding("test text")
        
        
        assert mock_client.embeddings.create.call_count == 2

@pytest.mark.asyncio
async def test_generate_embeddings_batch_max_retries_zero():
    """Test that max_retries=0 in batch raises RuntimeError."""
    embedder = EmbeddingGenerator(max_retries=0)
    
    with pytest.raises(RuntimeError, match="Failed to generate batch embeddings after 0 retries"):
        await embedder.generate_embeddings_batch(["test text"])

