"""
CLI entry point for cont3xt MCP server (stdio transport).

This runs the MCP server over stdio for compatibility with Cline, Claude Desktop,
and other MCP clients.

Usage:
  - Default (stdio):    virtual-context-mcp
  - With config path:   virtual-context-mcp --config configs/novel_writing.yaml
  - Initialize storage: virtual-context-mcp --init-db
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

from mcp.server.stdio import stdio_server

from .server import VirtualContextMCPServer


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="cont3xt MCP server (stdio)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (defaults to configs/novel_writing.yaml or env overrides)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize local storage (SQLite) and exit",
    )
    return parser.parse_args()


async def init_databases(config_path: Optional[str]) -> None:
    """Initialize SQLite-only storage for MVP."""
    app = VirtualContextMCPServer(config_path=config_path)
    await app.initialize()
    # Nothing else to do for MVP; no long-running connections to close.
    logging.getLogger(__name__).info("Storage initialization complete")


async def run_stdio(config_path: Optional[str]) -> None:
    """Run the MCP server over stdio."""
    app = VirtualContextMCPServer(config_path=config_path)
    server = await app.setup_server()

    # Start stdio transport and hand over to the server
    async with stdio_server() as (read, write):
        await server.run(read, write)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        if args.init_db:
            logger.info("Initializing storage...")
            asyncio.run(init_databases(args.config))
            logger.info("Initialization finished. Exiting.")
            return

        logger.info("Starting cont3xt MCP server (stdio)...")
        asyncio.run(run_stdio(args.config))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
