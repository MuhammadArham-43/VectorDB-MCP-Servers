import typing as T
from pydantic import Field
from pydantic_settings import BaseSettings

DEFAULT_TOOL_STORE_DESCRIPTION = (
    "Keep the memory for later use, when you are asked to remember something."
)

DEFAULT_TOOL_FIND_DESCRIPTION = (
    "Look up memories in Qdrant. Use this tool when you need to: \n"
    " - Find memories by their content \n"
    " - Access memories for further analysis \n"
    " - Get some personal information about the user \n"
)

class QdrantToolSettings(BaseSettings):
    """
    Configuration for all the tools
    """
    tool_store_description: str = Field(
        default=DEFAULT_TOOL_STORE_DESCRIPTION,
        validation_alias="TOOL_STORE_DESCRIPTION"
    )
    tool_find_description: str = Field(
        default=DEFAULT_TOOL_FIND_DESCRIPTION,
        validation_alias="TOOL_FIND_DESCRIPTION"
    )


class QdrantSettings(BaseSettings):
    """
    Configuration for the Qdrant connector
    """

    collection_name: str = Field(default=None, validation_alias="COLLECTION_NAME")
    local_path: T.Optional[str] = Field(default=None, validation_alias="QDRANT_LOCAL_PATH")
    location: T.Optional[str] = Field(default=None, validation_alias="QDRANT_URL")
    api_key: T.Optional[str] = Field(default=None, validation_alias="QDRANT_API_KEY")
    search_limit: int = Field(default=10, validation_alias="QDRANT_SEARCH_LIMIT")
    read_only: bool = Field(default=False, validation_alias="QDRANT_READ_ONLY")