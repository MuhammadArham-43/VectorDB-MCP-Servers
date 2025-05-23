import os
import time
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

from src.vectordb_mcp_servers.base_provider.base_mcp import Entry
from src.embeddings.base import EmbeddingProvider

class AOSSConnector:

    def __init__(
        self,
        host_url: str,
        aws_region: str,
        index_name: str,
        embedding_provider: EmbeddingProvider
    ) -> None:
        self.host_url = host_url.lstrip("https://")
        self.region = aws_region
        self.index_name = index_name
        self._embedding_provider = embedding_provider
        self.client = self._connect()
        assert self.client.indices.exists(index=self.index_name), f"Index {index_name} does not exist"
    
    def _connect(self):
        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials=credentials, region=self.region, service="aoss")
        return OpenSearch(
            hosts=[{"host": self.host_url, "port": 443}],
            http_auth = auth,
            use_ssl = True,
            verify_certs = True,
            connection_class = RequestsHttpConnection,
            pool_maxsize = 20
        )

    async def search(self, query: str, limit: int = 10):
        assert self.client.indices.exists(index=self.index_name), f"Index {self.index_name} does not exist"
        query_embedding = await self._embedding_provider.embed_query(query)

        search_body = {
            "size": limit,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": limit
                    }
                }
            }
        }
        response = self.client.search(index=self.index_name, body=search_body)
        return [Entry(content=hit["_source"]["text"], metadata=hit["_source"]["metadata"]) for hit in response["hits"]["hits"]]

