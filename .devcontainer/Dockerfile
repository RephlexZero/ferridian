FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    # --- build essentials ---
    build-essential \
    pkg-config \
    unzip \
    zip \
    # --- SSL ---
    libssl-dev \
    # --- Vulkan (runtime + dev headers + Mesa ICD + validation layers) ---
    libvulkan1 \
    libvulkan-dev \
    mesa-vulkan-drivers \
    vulkan-tools \
    vulkan-validationlayers \
    # --- X11 window-system client libraries (needed by winit on X11) ---
    libx11-dev \
    libxrandr-dev \
    libxi-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxext-dev \
    xauth \
    # --- OpenGL / EGL (Mesa software fallback + surface creation) ---
    libgl1-mesa-dri \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    mesa-utils \
    # --- Wayland (protocol headers, runtime, extra protocols) ---
    libwayland-dev \
    libwayland-client0 \
    wayland-protocols \
    # --- XKB (keyboard handling, required by winit) ---
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create the Wayland/XDG runtime directory used by containerEnv XDG_RUNTIME_DIR.
RUN mkdir -p /tmp/runtime-dir && chmod 0700 /tmp/runtime-dir
