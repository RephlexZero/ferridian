# The single container image (§4.3): golden-render CI job AND the devcontainer
# base, so dev and CI see the same rasteriser.
#
# Pinning model: the *pushed image* is the pin. .github/workflows/container.yml
# builds this file, records the resolved Mesa/VVL versions in the image
# (/etc/ferridian-pins.txt) and pushes an immutable date tag to GHCR; CI and
# baselines reference that tag. Bumping Mesa = rebuild + re-bless goldens in
# one deliberate PR (see goldens/README.md).

FROM ubuntu:24.04

ARG SLANG_VERSION=2026.13
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git git-lfs xz-utils unzip zip \
        build-essential pkg-config cmake \
        libvulkan1 libvulkan-dev vulkan-tools vulkan-validationlayers \
        mesa-vulkan-drivers \
        openjdk-21-jdk-headless \
        # X11/EGL client libs so a windowed engine can run against the host
        # display (real-GPU path via /dev/dri); lavapipe needs none of this.
        libx11-6 libxrandr2 libxi6 libxcursor1 libxinerama1 libxext6 xauth \
        libegl1 libgl1 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/* \
    && dpkg-query -W mesa-vulkan-drivers vulkan-validationlayers libvulkan1 \
        > /etc/ferridian-pins.txt \
    && cat /etc/ferridian-pins.txt

# Slang (slangc) — the pack compiler toolchain. Version must match mise.toml.
RUN curl -fsSL -o /tmp/slang.tar.gz \
        "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-x86_64.tar.gz" \
    && mkdir -p /opt/slang \
    && tar -xzf /tmp/slang.tar.gz -C /opt/slang \
    && rm /tmp/slang.tar.gz \
    && /opt/slang/bin/slangc -v

ENV PATH="/opt/slang/bin:${PATH}" \
    FERRIDIAN_SLANGC=/opt/slang/bin/slangc

# mise (task runner / tool pinning) system-wide.
RUN curl -fsSL https://mise.run | MISE_INSTALL_PATH=/usr/local/bin/mise sh

# rustup shared between root (CI) and the ubuntu user (devcontainer); the
# actual toolchain version comes from rust-toolchain.toml on first use.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH="/usr/local/cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain none --profile minimal --no-modify-path \
    # registry/ doesn't exist yet at this point; create it now so its
    # permissive mode is what gets copied into the named volume that mounts
    # over it later (an empty bind target created by Docker at run time would
    # otherwise land root-owned and unwritable by the ubuntu user).
    && mkdir -p "$CARGO_HOME/registry" \
    && chmod -R a+rwX "$RUSTUP_HOME" "$CARGO_HOME"

# Runtime dir some display plumbing expects when running windowed.
RUN mkdir -p /tmp/runtime-dir && chmod 1777 /tmp/runtime-dir
ENV XDG_RUNTIME_DIR=/tmp/runtime-dir

# ubuntu (uid 1000, present in this base image) owns its home dir's mise data
# subtree up front, mirroring the cargo/rustup trick above — otherwise the
# named volume the devcontainer mounts over it lands root-owned and the
# ubuntu user can't write to it without a one-off sudo chown.
RUN mkdir -p /home/ubuntu/.local/share/mise && chown -R ubuntu:ubuntu /home/ubuntu/.local
