import logging
import uuid
import typing as T

from qdrant_client import AsyncQdrantClient, models

from src.vectordb_mcp_servers.base_provider.base_mcp import Entry
from src.embeddings.base import EmbeddingProvider


class QdrantConnector:
    """
    Encapsulates the connection to the Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will required collection name to be provided.
    :param embedding_provider: The embedding provider to use
    :param qdrant_local_path: The path to storage directory for the Qdrant client, if local model is used.
    """

    def __init__(
            self,
            qdrant_url: T.Optional[str],
            qdrant_api_key: T.Optional[str],
            collection_name: T.Optional[str],
            embedding_provider: EmbeddingProvider,
            qdrant_local_path: T.Optional[str] = None
    ) -> None:
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=self._qdrant_url, api_key=self._qdrant_api_key, path=qdrant_local_path
        )
    
    async def get_collection_names(self) -> T.List[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]


    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            vector_size = self._embedding_provider.get_vector_size()
            vector_name = self._embedding_provider.get_vector_name()
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                }
            )
    
    async def store(self, entry: Entry, *, collection_name: T.Optional[str] = None):
        """
        Store some information in the Qdrant collection, along with specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in.
                                Optional. If not provided, default collection is used.
        """
        
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        await self._ensure_collection_exists(collection_name)

        embeddings = await self._embedding_provider.embed_documents([entry.content])

        vector_name = self._embedding_provider.get_vector_name()
        payload = {"document": entry.content, "metadata": entry.metadata}
        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={vector_name: embeddings[0]},
                    payload=payload
                )
            ]
        )
    
    async def search(self, query: str, *, collection_name: T.Optional[str] = None, limit: int = 10) -> T.List[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in. 
                                Optional. If not provided, default collection is used.
        :param limit: The maximum number of entries to return.
        :return: A list of entries found
        """
        
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists: return []

        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit
        )

        return [Entry(content=result.payload["document"], metadata=result.payload.get("metadata")) for result in search_results.points]