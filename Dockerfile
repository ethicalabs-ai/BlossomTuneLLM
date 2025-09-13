# This stage installs build tools, uv, and builds the Python virtual environment.
FROM python:3.11.12-slim AS builder

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y build-essential

# Install uv - a fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/

# Set the working directory
WORKDIR /app

# Create a virtual environment to keep dependencies isolated
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Install dependencies into the virtual environment using uv
# This is faster than pip and uses the lock file for reproducible builds.
RUN uv sync --locked --no-cache

# This stage creates the final, lean image for production.
FROM python:3.11.12-slim

# Copy the uv binary from the builder stage so it's available at runtime
COPY --from=builder /usr/local/bin/uv /usr/local/bin/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    # Add the virtual environment to the PATH
    PATH="/opt/venv/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
# This contains all the installed dependencies but not the build tools.
COPY --from=builder /opt/venv /opt/venv

# Copy your application code into the image
# Using .dockerignore prevents unnecessary files from being copied.
COPY . .

# Define the entrypoint and default command for the container
ENTRYPOINT ["/app/docker_entrypoint.sh"]
CMD ["superlink"]