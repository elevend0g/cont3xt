# Virtual Context MCP Server Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Download spaCy English model
RUN python -m spacy download en_core_web_sm

# Copy application source code
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY pyproject.toml .

# Install the application in development mode
RUN pip install -e .

# Create data directory for SQLite and other persistent storage
RUN mkdir -p /app/data

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose the default MCP server port
EXPOSE 8000

# Health check to ensure the service is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import virtual_context_mcp; print('OK')" || exit 1

# Default command to run the MCP server
CMD ["python", "-m", "virtual_context_mcp.server", "--config", "configs/novel_writing.yaml"]