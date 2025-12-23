from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from utils.logging_config import get_logger

logger = get_logger(__name__)


def load_environment() -> bool:
    """Load environment variables with robust path resolution."""
    script_dir = Path(__file__).resolve().parent

    # Possible .env file locations (in order of preference)
    possible_env_paths = [
        script_dir.parent / ".env",  # project root
        script_dir / ".env",  # utils/.env (rare but supported)
        Path.cwd() / ".env",  # current working directory
    ]

    for env_path in possible_env_paths:
        if env_path.exists():
            try:
                loaded = load_dotenv(env_path, override=True)
                if loaded:
                    logger.info("Loaded .env from: %s", env_path)
                else:
                    logger.warning(".env file found at %s but no variables loaded", env_path)
                return loaded
            except Exception as e:
                logger.warning("Error loading .env from %s: %s", env_path, e, exc_info=True)
                return False

    logger.warning("No .env file found. Please ensure .env exists with required variables")
    logger.info("Expected .env locations:")
    for p in possible_env_paths:
        logger.info("  - %s", p)
    return False


@dataclass(frozen=True)
class Settings:
    APP_NAME: str
    APP_VERSION: str


@lru_cache
def get_settings() -> Settings:
    # Ensure env is loaded (works even if you run from outside the repo root)
    load_environment()

    return Settings(
        APP_NAME=os.getenv("APP_NAME", "food_categorizer"),
        APP_VERSION=os.getenv("APP_VERSION", "0.1.0"),
    )


