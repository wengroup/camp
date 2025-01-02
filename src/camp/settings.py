from pydantic import Field
from pydantic_settings import BaseSettings


class CAMPSettings(BaseSettings):
    """Global settings."""

    DEBUG: bool = Field(False, description="Whether to run in debug mode.")
