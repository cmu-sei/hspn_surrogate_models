FROM nvcr.io/nvidia/pytorch:23.09-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    bash \
    nano \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://astral.sh/uv/install.sh | sh

WORKDIR /workspace

COPY pyproject.toml /workspace/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --cache-dir=/root/.cache/pip --upgrade pip && \
    pip install --cache-dir=/root/.cache/pip  -e .

RUN echo "root:password" | chpasswd
COPY --chmod=555 entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
