FROM python:3.10.16-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends bash coreutils ffmpeg libegl1 libgl1 libglib2.0-0 libgomp1 libgl1-mesa-dri && rm -rf /var/lib/apt/lists/*
# The full, independent Python 3.12 Newton/Warp prefix and fixed worker sources
# are mounted read-only from each task's frozen resource/source identities.
# The base-image Python is not used for scene execution.
RUN mkdir -p /opt/newton-env /opt/resources/perception /opt/worldrsi && printf '%s\n' '#!/bin/sh' 'export PYTHONPATH=/opt/resources/perception/.venv/lib/python3.12/site-packages' 'exec /opt/resources/perception/python/bin/python3.12 "$@"' > /usr/local/bin/perception-python && chmod 755 /usr/local/bin/perception-python
ENV PATH="/opt/newton-env/bin:${PATH}" PYTHONPATH=/opt/worldrsi PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
WORKDIR /work
CMD ["sleep", "infinity"]
