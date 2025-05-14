import typing as T
from src.embeddings.factory import create_embedding_provider
from src.vectordb_mcp_servers.base_provider.base_mcp import Entry, Metadata, BaseVectorDBMCPServer
from src.embeddings.types import EmbeddingProviderSettings
from src.vectordb_mcp_servers.aoss.aoss_connector import AOSSConnector
from src.vectordb_mcp_servers.aoss.settings import AossSettings, AossToolSettings


class AossMCPServer(BaseVectorDBMCPServer):

    def __init__(
        self,
        tool_settings: AossToolSettings,
        aoss_settings: AossSettings,
        embedding_provder_settings: EmbeddingProviderSettings,
        name: str  = "aoss-mcp-server", 
        instructions: str | None = None, 
        **settings: T.Any
    ):
        self.tool_settings = tool_settings
        self.aoss_settings = aoss_settings
        self.embedding_provder_settings = embedding_provder_settings
        self._name = name
        self.embedding_provider = create_embedding_provider(embedding_provder_settings)

        self.aoss_connector = AOSSConnector(
            host_url=aoss_settings.host_url,
            aws_region=aoss_settings.aws_region,
            index_name=aoss_settings.index_name,
            embedding_provider=self.embedding_provider
        )

        super().__init__(name, instructions, **settings)
    
    @property
    def name(self):
        return self._name


    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def find(query: str) -> str:
            """
            Find information in AWS OpenSearch Serverless.
            :param query: The query to use for search

            :return: A string of all relevant results. Contain xml-format entries with content and metadata.
            """
            entries = await self.aoss_connector.search(query)
            if not entries:
                return f"No information found for the query: {query}"
            content = [f"Results for the query: {query}"]
            for entry in entries:
                content.append(self.format_entry(entry))
            return "\n".join(content)

        self.add_tool(find, name="opensearch-find", description=self.tool_settings.tool_find_description)