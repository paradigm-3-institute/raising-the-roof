FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 curl ca-certificates git ripgrep && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /work && chmod 1777 /work

WORKDIR /work
