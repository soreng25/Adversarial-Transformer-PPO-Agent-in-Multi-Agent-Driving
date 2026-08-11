# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4

ARG DEBIAN_FRONTEND=noninteractive
ARG GPUDRIVE_REPOSITORY=https://github.com/Emerge-Lab/gpudrive.git
ARG GPUDRIVE_COMMIT=aa48a431ed127a37610cc2176db30ec73d0c55df
ARG GPUDRIVE_MADRONA_COMMIT=4bda33465340fabc2e61fb27f95aa04795a15466
ARG GPUDRIVE_JSON_COMMIT=0457de21cffb298c22b629e538036bfeb96130b7
ARG GPUDRIVE_UV_LOCK_SHA256=d4f6fe3df752ae9f5cf4fd3e8c510870273fefc4e3c0f3df401f16a62ef7a04d
ARG GPUDRIVE_SCENE_SHA256=69bd2b9ae49d43745651262abf3956309e9c0092ca24aff72e0f9abb32f9b948
ARG UV_WHEEL_URL=https://files.pythonhosted.org/packages/8d/c4/97fdd4fca11d06633bb500849f70e4e6b201bcba3833894732e709be2d60/uv-0.12.3-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
ARG UV_WHEEL_SHA256=1482d1462b1aecd18ee33627363fe1c63d6a194f12d40d37efc446d9e0d800a1
ARG PORT_GIT_COMMIT
ARG PORT_DIRTY
ARG PORT_DIFF_SHA256
ARG PORT_SOURCE_TREE_SHA256

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libjpeg-dev \
        libpng-dev \
        libx11-dev \
        libxcursor-dev \
        libxi-dev \
        libxinerama-dev \
        libxrandr-dev \
        libc++1 \
        mesa-common-dev \
        ninja-build \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl --fail --location --silent --show-error "${UV_WHEEL_URL}" -o /tmp/uv.whl \
    && echo "${UV_WHEEL_SHA256}  /tmp/uv.whl" | sha256sum --check --strict \
    && python3 -m pip install --no-cache-dir /tmp/uv.whl \
    && test "$(uv --version)" = "uv 0.12.3" \
    && rm /tmp/uv.whl

RUN git clone --filter=blob:none --no-checkout "${GPUDRIVE_REPOSITORY}" /opt/gpudrive \
    && git -C /opt/gpudrive fetch --depth=1 origin "${GPUDRIVE_COMMIT}" \
    && git -C /opt/gpudrive checkout --detach "${GPUDRIVE_COMMIT}" \
    && git -C /opt/gpudrive submodule update --init --recursive --depth=1 \
    && test "$(git -C /opt/gpudrive rev-parse HEAD)" = "${GPUDRIVE_COMMIT}" \
    && test "$(git -C /opt/gpudrive/external/madrona rev-parse HEAD)" = "${GPUDRIVE_MADRONA_COMMIT}" \
    && test "$(git -C /opt/gpudrive/external/json rev-parse HEAD)" = "${GPUDRIVE_JSON_COMMIT}" \
    && echo "${GPUDRIVE_UV_LOCK_SHA256}  /opt/gpudrive/uv.lock" | sha256sum --check --strict \
    && echo "${GPUDRIVE_SCENE_SHA256}  /opt/gpudrive/data/processed/examples/tfrecord-00000-of-01000_325.json" | sha256sum --check --strict

WORKDIR /opt/gpudrive
RUN uv sync --frozen --python 3.11.9 \
    && test "$(.venv/bin/python -c 'import platform; print(platform.python_version())')" = "3.11.9"

RUN mkdir -p build \
    && cd build \
    && ../.venv/bin/cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/ ../.venv/bin/cmake --build . --parallel "$(nproc)" \
    && rm /usr/local/cuda/lib64/stubs/libcuda.so.1

COPY . /workspace/port
WORKDIR /workspace/port

ENV PATH=/opt/gpudrive/.venv/bin:${PATH}
ENV PYTHONPATH=/workspace/port/src:/opt/gpudrive:/opt/gpudrive/build
ENV MADRONA_MWGPU_KERNEL_CACHE=/var/cache/gpudrive/megakernel.bin
ENV GPUDRIVE_REFERENCE_IMAGE_DIGEST=sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4
ENV GPUDRIVE_PORT_GIT_COMMIT=${PORT_GIT_COMMIT}
ENV GPUDRIVE_PORT_DIRTY=${PORT_DIRTY}
ENV GPUDRIVE_PORT_DIFF_SHA256=${PORT_DIFF_SHA256}
ENV GPUDRIVE_PORT_SOURCE_TREE_SHA256=${PORT_SOURCE_TREE_SHA256}
ENV GPUDRIVE_SOURCE=/opt/gpudrive
ENV GPUDRIVE_VICTIM_CHECKPOINT=/opt/checkpoints/policy_S10_000_02_27/1532950cad84dafc6e9d976a2bcc524ee481a1a1

RUN python scripts/fetch_victim_checkpoint.py \
        --destination "${GPUDRIVE_VICTIM_CHECKPOINT}" \
        --gpudrive-source /opt/gpudrive

LABEL org.opencontainers.image.source="https://github.com/soreng25/Adversarial-Transformer-PPO-Agent-in-Multi-Agent-Driving"
LABEL org.opencontainers.image.gpudrive.commit="aa48a431ed127a37610cc2176db30ec73d0c55df"

CMD ["python", "-m", "gpudrive_adversary", "doctor", "--source", "/opt/gpudrive", "--reference", "--strict"]
