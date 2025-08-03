# Install uv
FROM python:3.11.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install build-essential -y

# Change the working directory to the `app` directory
WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock,relabel=shared \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml,relabel=shared \
    uv sync --locked --no-install-project

# Copy the project into the image
ADD . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

ENTRYPOINT ["/app/docker_entrypoint.sh"]
CMD ["superlink"]