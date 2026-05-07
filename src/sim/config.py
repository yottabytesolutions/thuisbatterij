"""Runtime-configuratie afgeleid uit gebruikersconfiguratie."""


from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

from .userconfig import UserConfig


class Settings(BaseSettings):
    """Algemene instellingen voor runtime-gebruik.

    Pydantic-settings leest velden ook uit env vars (case-insensitive). De
    ENTSO-E API-key hoort in `ENTSOE_API_KEY`, niet in een gedeeld config-bestand.
    """

    questdb_url: str = "http://localhost:9000"
    entsoe_api_key: str | None = None
    entsoe_zone: str = "NL"
    cache_dir: Path = Field(default_factory=lambda: Path("cache"))
    output_dir: Path = Field(default_factory=lambda: Path("output"))


def load_settings(user: UserConfig | None = None) -> Settings:
    """Bouw `Settings` met user-config als override op env-defaults.

    Lege of niet-gezette TOML-waarden tellen niet als override, zodat env vars
    en class-defaults zichtbaar blijven. Een lege `entsoe_api_key` of
    `entsoe_zone` zou anders een werkende env-default of `"NL"`-default
    overschrijven.
    """
    kwargs: dict[str, object] = {}
    if user is not None:
        if user.simulation.questdb_url:
            kwargs["questdb_url"] = user.simulation.questdb_url
        if user.simulation.entsoe_zone:
            kwargs["entsoe_zone"] = user.simulation.entsoe_zone
        if user.simulation.entsoe_api_key:
            kwargs["entsoe_api_key"] = user.simulation.entsoe_api_key
    s = Settings(**kwargs)
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    s.output_dir.mkdir(parents=True, exist_ok=True)
    return s
