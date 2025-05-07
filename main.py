from src.qdrant_mcp_server.settings import QdrantSettings, ToolSettings, EmbeddingProviderSettings
from src.qdrant_mcp_server.mcp_server import QdrantMCPServer



mcp = QdrantMCPServer(
    qdrant_settings=QdrantSettings(),
    tool_settings=ToolSettings(),
    embedding_provider_settings=EmbeddingProviderSettings()
)

if __name__ == "__main__":
    print(f"Starting {mcp.name} on stdio...")
    mcp.run(transport="stdio")
    

