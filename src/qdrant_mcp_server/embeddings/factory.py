from src.qdrant_mcp_server.embeddings.base import EmbeddingProvider
from src.qdrant_mcp_server.embeddings.types import EmbeddingProviderType
from src.qdrant_mcp_server.settings import EmbeddingProviderSettings

def create_embedding_provider(settings: EmbeddingProviderSettings) -> EmbeddingProvider:
    """
    Create an embedding provider based on the specified type.
    :param settings: The settings for the embedding provider
    :return: An instance of the specified embedding provider
    """
    if settings.provider_type == EmbeddingProviderType.FASTEMBED:
        from src.qdrant_mcp_server.embeddings.fastembed_provider import FastEmbedProvider
        return FastEmbedProvider(settings.model_name)
    raise ValueError(f"Unsupported embedding provider: {settings.provider_type}")