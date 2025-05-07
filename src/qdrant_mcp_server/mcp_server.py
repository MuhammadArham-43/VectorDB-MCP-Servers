import json
import logging
import typing as T

from mcp.server.fastmcp import Context, FastMCP

from src.qdrant_mcp_server.embeddings.factory import create_embedding_provider
from src.qdrant_mcp_server.qdrant import Entry, Metadata, QdrantConnector
from src.qdrant_mcp_server.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings
)

logger = logging.getLogger(__name__)

class QdrantMCPServer(FastMCP):
    """
    MCP Server for Qdrant
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
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
        self.qdrant_connector = QdrantConnector(
            qdrant_url=qdrant_settings.location,
            qdrant_api_key=qdrant_settings.api_key,
            collection_name=qdrant_settings.collection_name,
            embedding_provider=self.embedding_provider,
            qdrant_local_path=qdrant_settings.local_path
        )

        super().__init__(name=name, instructions=instructions, **settings)
        self.setup_tools()
    
    @property
    def name(self):
        return self._name

    def format_entry(self, entry: Entry) -> str:
        """
        Formats the Entry into a string description.
        Override this in a subclass to customize the format.
        """

        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content.strip()}</content><metadata>{entry_metadata}</metadata></entry>"

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            ctx: Context,
            information: str,
            collection_name: str,
            metadata: Metadata = None
        ) -> str:
            """
            Store some information in Qdrant
            :param ctx: The context of the request
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information [Optional].
            :param collection_name: The name of the collection to store in.
                                    [Optional]. Default collection is used if not provided.
            :return: A message indicating the information that was stored.
            """
            await ctx.debug(f"Storing information {information} in Qdrant")

            entry = Entry(content=information, metadata=metadata)
            await self.qdrant_connector.store(entry, collection_name=collection_name)
            if collection_name:
                return f"Stored: {information} in collection {collection_name}"
            return f"Stored: {information}"

        async def store_with_default_collection(
            ctx: Context,
            information: str,
            metadata: Metadata = None
        ) -> str:
            assert self.qdrant_settings.collection_name is not None
            return await store(
                ctx, information, self.qdrant_settings.collection_name, metadata
            )
        
        async def find(
            ctx: Context,
            query: str,
            collection_name: str
        ) -> str:
            """
            Find memories in Qdrant.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :return: A string of all relevant results. Contain xml-format entries with content and metadata.
            """

            await ctx.debug(f"Finding results for query: {query}")
            if collection_name: await ctx.debug(f"Overriding the collection name with {collection_name}")
            
            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
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
    
        async def find_with_default_collection(
            ctx: Context,
            query: str
        ) -> T.List[str]:
            assert self.qdrant_settings.collection_name is not None
            return await find(ctx, query, self.qdrant_settings.collection_name)
        
        if self.qdrant_settings.collection_name:
            self.add_tool(
                find_with_default_collection,
                name="qdrant-find",
                description=self.tool_settings.tool_find_description
            )
        else:
            self.add_tool(
                find,
                name="qdrant-find",
                description=self.tool_settings.tool_find_description
            )
        
        if not self.qdrant_settings.read_only:
            if self.qdrant_settings.collection_name:
                self.add_tool(
                    store_with_default_collection,
                    name="qdrant-store",
                    description=self.tool_settings.tool_store_description
                )
            else:
                self.add_tool(
                    store,
                    name="qdrant-store",
                    description=self.tool_settings.tool_store_description
                )            