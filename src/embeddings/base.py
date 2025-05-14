from abc import ABC, abstractmethod
from typing import List

class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers
    """

    @abstractmethod
    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed a list of documents into vectors
        """
        raise NotImplementedError("Subclasses must override embed_documents method")

    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a query into a vector
        """
        raise NotImplementedError("Subclasses must override embed_query method")

    @abstractmethod
    def get_vector_name(self) -> str:
        """
        Get the name of the vector for the Qdrant collection
        """
        raise NotImplementedError("Subclasses must override get_vector_name method")

    @abstractmethod
    def get_vector_size(self) -> str:
        """
        Get the size of the embedding vector for the Qdrant collection
        """
        raise NotImplementedError("Subclasses must override get_vector_size method")