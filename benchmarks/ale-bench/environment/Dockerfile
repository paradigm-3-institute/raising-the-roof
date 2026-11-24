# The agents' image: ALE-Bench's execution image for the task's language (filled in per
# task; AtCoder's 2023 language set), so programs compile and run here exactly as the
# judge compiles and runs them.
FROM {base_image}

# The agent CLI's setup asks apt for these; they are installed here and the apt sources
# removed, so setup needs no network (the task allows none but the model API).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git ripgrep python3 && \
    rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/* /etc/apt/sources.list

WORKDIR /work
