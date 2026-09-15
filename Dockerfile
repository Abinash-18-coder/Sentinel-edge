# SentinelEdge Production Dockerfile
FROM python:3.12-slim-bookworm

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install native OS dependencies required by OpenCV and ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and packaging tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir reportlab

# Copy application source code, models, and configuration
COPY configs/ ./configs/
COPY models/ ./models/
COPY src/ ./src/
COPY ui/ ./ui/

# Expose FastAPI HTTP & WebSocket port
EXPOSE 8000

# Launch production server via Uvicorn
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]