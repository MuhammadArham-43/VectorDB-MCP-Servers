import typing as T
from src.embeddings.factory import create_embedding_provider
from src.vectordb_mcp_servers.base_provider.base_mcp import Entry, Metadata, BaseVectorDBMCPServer
from src.vectordb_mcp_servers.qdrant_mcp_server.qdrant import QdrantConnector
from src.vectordb_mcp_servers.qdrant_mcp_server.settings import QdrantSettings, QdrantToolSettings
from src.embeddings.types import EmbeddingProviderSettings


class QdrantMCPServer(BaseVectorDBMCPServer):
    """
    MCP Server for Qdrant
    """

    def __init__(
        self,
        tool_settings: QdrantToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: EmbeddingProviderSettings,
        name: str = "qdrant-mcp-server",
        instructions: str | None = None,
        **settings: T.Any
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings
        self.embedding_provider_settigns = embedding_provider_settings
        self._name = name
        self.embedding_provider = create_embedding_provider(embedding_provider_settings)
        
        self.collection_name = qdrant_settings.collection_name
        assert self.collection_name is not None and self.collection_name != "", "Must provide a collection name" 
        
        self.qdrant_connector = QdrantConnector(
            qdrant_url=qdrant_settings.location,
            qdrant_api_key=qdrant_settings.api_key,
            collection_name=qdrant_settings.collection_name,
            embedding_provider=self.embedding_provider,
            qdrant_local_path=qdrant_settings.local_path
        )

        super().__init__(name=name, instructions=instructions, **settings)
    
    @property
    def name(self):
        return self._name


    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            information: str,
            metadata: Metadata = None
        ) -> str:
            """
            Store some information in Qdrant
            :param ctx: The context of the request
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information [Optional].

            :return: A message indicating the information that was stored.
            """
            entry = Entry(content=information, metadata=metadata)
            await self.qdrant_connector.store(entry, collection_name=self.collection_name)
            if self.collection_name:
                return f"Stored: {information} in collection {self.collection_name}"
            return f"Stored: {information}"
        
        async def find(
            query: str,
        ) -> str:
            """
            Find memories in Qdrant.
            :param query: The query to use for the search.

            :return: A string of all relevant results. Contain xml-format entries with content and metadata.
            """
            entries = await self.qdrant_connector.search(
                query,
                collection_name=self.collection_name,
                limit=self.qdrant_settings.search_limit
            )

            if not entries:
                return [f"No information found for the query: '{query}'"]
            
            content = [
                f"Results for the query: '{query}'"
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return "\n".join(content)
        
        self.add_tool(
            find,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description
        )
        
        if not self.qdrant_settings.read_only:
            self.add_tool(
                store,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description
            )            