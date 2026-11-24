# The search filter's relay (browsecomp_task.py RELAY_IMAGE): proxy.py's Python and libraries, nothing else.
# The code, blocklist and protected texts are mounted at run time. Build: docker build -t ttc-browsecomp-relay:1 .
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
COPY relay-requirements.txt /tmp/relay-requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends zstd && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --no-deps -r /tmp/relay-requirements.txt && rm /tmp/relay-requirements.txt
