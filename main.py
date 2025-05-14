from typing import Literal
from src.embeddings.types import EmbeddingProviderSettings
from src.vectordb_mcp_servers.base_provider.settings import ProviderSettings, ValidProviders

def get_mcp(provider: ValidProviders):
    if provider == "QDRANT":
        from src.vectordb_mcp_servers.qdrant_mcp_server.settings import QdrantSettings, QdrantToolSettings
        from src.vectordb_mcp_servers.qdrant_mcp_server.mcp_server import QdrantMCPServer
        return QdrantMCPServer(
            qdrant_settings=QdrantSettings(),
            tool_settings=QdrantToolSettings(),
            embedding_provider_settings=EmbeddingProviderSettings()
        )
    if provider == "AOSS":
        from src.vectordb_mcp_servers.aoss.settings import AossSettings, AossToolSettings
        from src.vectordb_mcp_servers.aoss.aoss_mcp import AossMCPServer
        return AossMCPServer(
            tool_settings=AossToolSettings(),
            aoss_settings=AossSettings(),
            embedding_provder_settings=EmbeddingProviderSettings()
        )
    raise ValueError(f"Provider {provider} not implemented.")

if __name__ == "__main__":
    provider = ProviderSettings().provider_name
    mcp = get_mcp(provider)
    print(f"Starting {mcp.name} on stdio...")
    mcp.run(transport="stdio")
    

