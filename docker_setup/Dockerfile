# NVIDIA Isaac ROS base — ROS2 Humble + CUDA + cuDNN + TensorRT + PyTorch + Isaac ROS pkgs
#
# Default tag below is x86_64 (for WSL desktop dev).  On Jetson, override with:
#   docker build --build-arg ROS_BASE=nvcr.io/nvidia/isaac/ros:aarch64-ros2_humble ...
# or set ROS_BASE in the docker compose environment.
ARG ROS_BASE=nvcr.io/nvidia/isaac/ros:x86_64-ros2_humble
FROM ${ROS_BASE}

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# ── Extra ROS packages we use that may not be in the Isaac ROS base image ─
RUN apt update && apt install -y \
    python3-colcon-common-extensions \
    ros-humble-backward-ros \
    ros-humble-camera-info-manager \
    ros-humble-cv-bridge \
    ros-humble-diagnostic-updater \
    ros-humble-image-publisher \
    ros-humble-image-transport \
    ros-humble-rclcpp-components \
    ros-humble-tf2-sensor-msgs \
    libgflags-dev \
    nlohmann-json3-dev \
    libgoogle-glog-dev \
    libssl-dev \
    python3-opencv \
    usbutils \
    && rm -rf /var/lib/apt/lists/*

# ── Desktop environment + VNC server ──────────────────────────────────────
RUN apt update && \
    echo "keyboard-configuration keyboard-configuration/layoutcode string us" | debconf-set-selections && \
    echo "keyboard-configuration keyboard-configuration/modelcode string pc105" | debconf-set-selections && \
    apt install -y \
    xfce4 \
    xfce4-goodies \
    tigervnc-standalone-server \
    tigervnc-common \
    dbus-x11 \
    x11-xserver-utils \
    && rm -rf /var/lib/apt/lists/*

# ── ROS visualization tools ───────────────────────────────────────────────
RUN apt update && apt install -y \
    ros-humble-rqt \
    ros-humble-rqt-common-plugins \
    ros-humble-rqt-image-view \
    ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# VNC setup
RUN mkdir -p ~/.vnc && \
    echo "octane" | vncpasswd -f > ~/.vnc/passwd && \
    chmod 600 ~/.vnc/passwd

RUN mkdir -p /root/.vnc && \
    echo '#!/bin/bash' > /root/.vnc/xstartup && \
    echo 'unset SESSION_MANAGER' >> /root/.vnc/xstartup && \
    echo 'unset DBUS_SESSION_BUS_ADDRESS' >> /root/.vnc/xstartup && \
    echo 'exec startxfce4' >> /root/.vnc/xstartup && \
    chmod +x /root/.vnc/xstartup

# Desktop helper scripts
RUN mkdir -p /root/Desktop && \
    echo '#!/bin/bash' > /root/Desktop/start_rqt_image_view.sh && \
    echo 'source /opt/ros/humble/setup.bash' >> /root/Desktop/start_rqt_image_view.sh && \
    echo 'if [ -f /workspace/install/setup.bash ]; then' >> /root/Desktop/start_rqt_image_view.sh && \
    echo '    source /workspace/install/setup.bash' >> /root/Desktop/start_rqt_image_view.sh && \
    echo 'fi' >> /root/Desktop/start_rqt_image_view.sh && \
    echo 'ros2 run rqt_image_view rqt_image_view' >> /root/Desktop/start_rqt_image_view.sh && \
    chmod +x /root/Desktop/start_rqt_image_view.sh && \
    echo '#!/bin/bash' > /root/Desktop/start_rviz.sh && \
    echo 'source /opt/ros/humble/setup.bash' >> /root/Desktop/start_rviz.sh && \
    echo 'if [ -f /workspace/install/setup.bash ]; then' >> /root/Desktop/start_rviz.sh && \
    echo '    source /workspace/install/setup.bash' >> /root/Desktop/start_rviz.sh && \
    echo 'fi' >> /root/Desktop/start_rviz.sh && \
    echo 'ros2 run rviz2 rviz2' >> /root/Desktop/start_rviz.sh && \
    chmod +x /root/Desktop/start_rviz.sh

# ── git (needed for DA3 install from source) ──────────────────────────────
RUN apt update && apt install -y git && rm -rf /var/lib/apt/lists/*

# ── Depth Anything V3 dependencies ────────────────────────────────────────
# Isaac ROS image already includes CUDA-built torch + torchvision, so we
# skip those. Just install the small Python deps + DA3 itself.
RUN pip3 install --no-cache-dir \
        einops \
        huggingface_hub \
        safetensors \
        omegaconf \
        trimesh \
        open3d \
        imageio \
        "numpy<2" \
    && pip3 install --no-cache-dir git+https://github.com/ByteDance-Seed/Depth-Anything-3.git

WORKDIR /workspace

# Expose VNC port
EXPOSE 5901
