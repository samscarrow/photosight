# Photosight API Dockerfile
# Privacy-first computer vision service

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY photosight/ ./photosight/
COPY api/ ./api/

# Create non-root user for security
RUN useradd -m -u 1000 photosight && \
    chown -R photosight:photosight /app
USER photosight

# Set privacy-first environment defaults
ENV ENABLE_TELEMETRY=false
ENV STORE_TEMP_FILES=false
ENV MAX_IMAGE_SIZE_MB=10
ENV PYTHONPATH=/app

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]