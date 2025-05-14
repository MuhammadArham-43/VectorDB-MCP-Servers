from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings


ValidProviders = Literal["QDRANT", "AOSS"]

class ProviderSettings(BaseSettings):
    provider_name: ValidProviders = Field(default=None, validation_alias="VECTORDB_PROVIDER")