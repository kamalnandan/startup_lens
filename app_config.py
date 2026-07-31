"""Application configuration loaded from the project .env or Azure App Settings."""

import os
from pathlib import Path

from dotenv import dotenv_values


ENV_PATH = Path(__file__).resolve().with_name(".env")
_LOCAL_SETTINGS = dotenv_values(ENV_PATH) if ENV_PATH.is_file() else None


def get_required_setting(name: str) -> str:
    """Read a required setting from .env locally, or the process environment on Azure."""
    if _LOCAL_SETTINGS is not None:
        value = _LOCAL_SETTINGS.get(name)
        source = str(ENV_PATH)
    else:
        value = os.environ.get(name)
        source = "the process environment"

    if not value:
        raise RuntimeError(f"Required setting {name!r} is missing from {source}.")
    return value


def get_setting(name: str, default: str) -> str:
    """Read an optional setting from the same single configuration source."""
    if _LOCAL_SETTINGS is not None:
        return _LOCAL_SETTINGS.get(name) or default
    return os.environ.get(name) or default
