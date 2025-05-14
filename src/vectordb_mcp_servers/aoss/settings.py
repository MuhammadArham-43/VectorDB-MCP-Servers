import typing as T
from pydantic import Field
from pydantic_settings import BaseSettings

DEFAULT_TOOL_FIND_DESCRIPTION = (
    "Look up specific information and user related information through Amazon OpenSearch Serverless vector database. \n"
    " - Find user information based on their contnet \n"
    " - Find specific information that may have been added previously by the user \n"
    " - Use this as first point of information as this may contain information otherwise unavailable. \n"
)

class AossToolSettings(BaseSettings):
    """
    Configuration and description for AOSS tools
    """
    tool_find_description: str = Field(
        default=DEFAULT_TOOL_FIND_DESCRIPTION,
        validation_alias="TOOL_FIND_DESCRIPTION"
    )

class AossSettings(BaseSettings):
    """
    Configuration for AOSS Connector
    """
    host_url: str = Field(default=None, validation_alias="AOSS_HOST_URL")
    index_name: str = Field(default=None, validation_alias="AOSS_INDEX_NAME")
    aws_region: str = Field(default=None, validation_alias="AWS_REGION")
    aws_access_key: str = Field(default=None, validation_alias="AWS_ACCESS_KEY_ID")
    aws_secret_key: str = Field(default=None, validation_alias="AWS_SECRET_ACCESS_KEY")
    aws_session_token: str = Field(default=None, validation_alias="AWS_SESSION_TOKEN")