# Use official Python runtime as base image
# python:3.11.7-slim is lightweight and has Python 3.11.7 pre-installed
FROM python:3.11.7-slim

# Set metadata for the image
LABEL maintainer="k.nyiringan@alustudent.com"
LABEL description="FastAPI ML Prediction Service"

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for your packages
# These are required for numpy, opencv, and other scientific packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgfortran5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file first (for better caching)
# Docker caches layers, so if requirements don't change, 
# it won't reinstall packages
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Create a non-root user for security
# Running as root in containers is a security risk
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port your app runs on
EXPOSE 8000

# Add health check endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run when container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]