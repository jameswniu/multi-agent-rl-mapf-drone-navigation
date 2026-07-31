# Minimal Dockerfile for running the project
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Install the project itself so the src/ layout is importable
# (main, integrity_validators, env, agents, ...) exactly as in CI.
RUN pip install --no-cache-dir -e .

# Default command runs training
CMD ["python", "-m", "main"]
