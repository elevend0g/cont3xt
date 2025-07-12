"""Configuration management for virtual context MCP."""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class ContextConfig(BaseModel):
    """Configuration for context management."""
    max_tokens: int = 12000
    pressure_threshold: float = 0.8
    relief_percentage: float = 0.4
    chunk_size: int = 3200
    token_model: str = "cl100k_base"


class DatabaseConfig(BaseModel):
    """Configuration for database connections."""
    sqlite_path: str = "./data/memory.db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "story_memory"
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"


class Config(BaseModel):
    """Main configuration class."""
    context: ContextConfig = ContextConfig()
    database: DatabaseConfig = DatabaseConfig()
    model_name: str = "all-MiniLM-L6-v2"
    debug: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()  # Return default config if file doesn't exist
        
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        
        return cls(**yaml_data)

    @classmethod
    def from_env(cls, base_config: Optional["Config"] = None) -> "Config":
        """Override configuration with environment variables."""
        if base_config is None:
            base_config = cls()
        
        # Context config overrides
        context_data = base_config.context.model_dump()
        context_data["max_tokens"] = int(os.getenv("CONTEXT_MAX_TOKENS", context_data["max_tokens"]))
        context_data["pressure_threshold"] = float(os.getenv("CONTEXT_PRESSURE_THRESHOLD", context_data["pressure_threshold"]))
        context_data["relief_percentage"] = float(os.getenv("CONTEXT_RELIEF_PERCENTAGE", context_data["relief_percentage"]))
        context_data["chunk_size"] = int(os.getenv("CONTEXT_CHUNK_SIZE", context_data["chunk_size"]))
        context_data["token_model"] = os.getenv("CONTEXT_TOKEN_MODEL", context_data["token_model"])
        
        # Database config overrides
        database_data = base_config.database.model_dump()
        database_data["sqlite_path"] = os.getenv("DATABASE_SQLITE_PATH", database_data["sqlite_path"])
        database_data["qdrant_url"] = os.getenv("DATABASE_QDRANT_URL", database_data["qdrant_url"])
        database_data["qdrant_collection"] = os.getenv("DATABASE_QDRANT_COLLECTION", database_data["qdrant_collection"])
        database_data["neo4j_url"] = os.getenv("DATABASE_NEO4J_URL", database_data["neo4j_url"])
        database_data["neo4j_user"] = os.getenv("DATABASE_NEO4J_USER", database_data["neo4j_user"])
        database_data["neo4j_password"] = os.getenv("DATABASE_NEO4J_PASSWORD", database_data["neo4j_password"])
        
        # Main config overrides
        model_name = os.getenv("MODEL_NAME", base_config.model_name)
        debug = os.getenv("DEBUG", str(base_config.debug)).lower() in ("true", "1", "yes", "on")
        
        return cls(
            context=ContextConfig(**context_data),
            database=DatabaseConfig(**database_data),
            model_name=model_name,
            debug=debug
        )

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration from YAML file with environment variable overrides."""
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH", "configs/novel_writing.yaml")
        
        # Load from YAML
        base_config = cls.from_yaml(config_path)
        
        # Apply environment overrides
        return cls.from_env(base_config)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file path or default location."""
    return Config.load(config_path)