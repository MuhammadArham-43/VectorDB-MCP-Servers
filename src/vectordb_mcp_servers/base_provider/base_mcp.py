import json
import typing as T
from abc import abstractmethod
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP



Metadata = T.Dict[str, T.Any]
class Entry(BaseModel):
    """
    A single entry in the Qdrant collection
    """
    content: str
    metadata: T.Optional[Metadata] = None


class BaseVectorDBMCPServer(FastMCP):

    def __init__(self, name: str | None = None, instructions: str | None = None, **settings: T.Any):
        super().__init__(name, instructions, **settings)
        self.setup_tools()

    @property
    @abstractmethod
    def name(self):
        pass

    def format_entry(self, entry: Entry) -> str:
        """
        Formats the Entry into a string description.
        Override this in a subclass to customize the format.
        """

        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content.strip()}</content><metadata>{entry_metadata}</metadata></entry>"

    @abstractmethod
    async def setup_tools(self):
        pass