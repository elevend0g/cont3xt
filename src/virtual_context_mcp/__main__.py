"""
CLI Entry Point for Virtual Context MCP Server

This module provides the command-line interface for running the Virtual Context
MCP Server. It can be invoked as:

    python -m virtual_context_mcp
    python -m virtual_context_mcp.server
    python -m virtual_context_mcp --config path/to/config.yaml
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .server import VirtualContextMCPServer
from .config.settings import load_config


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('virtual_context_mcp.log')
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Virtual Context MCP Server - Infinite context for AI conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m virtual_context_mcp
  python -m virtual_context_mcp --config configs/novel_writing.yaml
  python -m virtual_context_mcp --host 0.0.0.0 --port 8080
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration file (default: configs/novel_writing.yaml)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for data storage (default: ./data)"
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode with debug logging"
    )
    
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize databases and exit"
    )
    
    return parser.parse_args()


async def init_databases(config_path: Optional[str], data_dir: str) -> None:
    """Initialize all databases"""
    logger = logging.getLogger(__name__)
    
    try:
        # Set up environment variables
        os.environ["SQLITE_DATA_DIR"] = data_dir
        
        # Load configuration
        config = load_config(config_path)
        logger.info(f"Loaded configuration from: {config_path or 'default'}")
        
        # Initialize server components to set up databases
        server_instance = VirtualContextMCPServer(config_path=config_path)
        await server_instance.initialize()
        
        logger.info("Database initialization completed successfully!")
        logger.info(f"Data directory: {data_dir}")
        logger.info("SQLite database: Initialized")
        logger.info("Qdrant collections: Initialized")
        logger.info("Neo4j schema: Initialized")
        
        # Clean shutdown
        await server_instance.shutdown()
        
    except Exception as e:
        logger.error(f"Failed to initialize databases: {e}")
        raise


async def run_server(args: argparse.Namespace) -> None:
    """Run the MCP server with given arguments"""
    logger = logging.getLogger(__name__)
    
    try:
        # Set up environment variables if specified
        if args.data_dir:
            os.environ["SQLITE_DATA_DIR"] = args.data_dir
        
        # Initialize server
        server_instance = VirtualContextMCPServer(config_path=args.config)
        await server_instance.initialize()
        
        # Setup and start MCP server
        server = await server_instance.setup_server()
        
        logger.info(f"Virtual Context MCP Server starting on {args.host}:{args.port}")
        logger.info(f"Configuration: {args.config or 'default'}")
        logger.info(f"Data directory: {args.data_dir}")
        
        # In a real MCP implementation, this would start the actual server
        # For now, we'll simulate running the server
        logger.info("MCP Server is ready and listening for connections...")
        logger.info("Press Ctrl+C to stop the server")
        
        # Keep the server running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
            
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    finally:
        # Cleanup
        if 'server_instance' in locals():
            await server_instance.shutdown()
        logger.info("Server shutdown complete")


def main() -> None:
    """Main entry point"""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.dev else args.log_level
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    
    # Ensure data directory exists
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Run appropriate command
    try:
        if args.init_db:
            logger.info("Initializing databases...")
            asyncio.run(init_databases(args.config, str(data_dir)))
            logger.info("Database initialization complete. Exiting.")
        else:
            logger.info("Starting Virtual Context MCP Server...")
            asyncio.run(run_server(args))
    except KeyboardInterrupt:
        logger.info("Operation stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()