"""Runtime-configuratie afgeleid uit gebruikersconfiguratie."""


from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

from .userconfig import UserConfig


class Settings(BaseSettings):
    """Algemene instellingen voor runtime-gebruik."""

    questdb_url: str = "http://localhost:9000"
    entsoe_api_key: str | None = None
    cache_dir: Path = Field(default_factory=lambda: Path("cache"))
    output_dir: Path = Field(default_factory=lambda: Path("output"))


def load_settings(user: UserConfig | None = None) -> Settings:
    kwargs = {}
    if user is not None:
        kwargs["questdb_url"] = user.simulation.questdb_url
        kwargs["entsoe_api_key"] = user.simulation.entsoe_api_key
    s = Settings(**kwargs)
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    s.output_dir.mkdir(parents=True, exist_ok=True)
    return s
