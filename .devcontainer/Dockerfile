FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libvulkan-dev \
    libwayland-dev \
    libxkbcommon-dev \
    mesa-vulkan-drivers \
    pkg-config \
    unzip \
    vulkan-tools \
    zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
