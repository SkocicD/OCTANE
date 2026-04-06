FROM ros:humble

SHELL ["/bin/bash", "-c"]

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install ROS dependencies and desktop environment
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

# Install desktop environment and VNC server
# Pre-configure keyboard layout to avoid prompts
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

# Install ROS visualization tools
RUN apt update && apt install -y \
    ros-humble-rqt \
    ros-humble-rqt-common-plugins \
    ros-humble-rqt-image-view \
    ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# Set up VNC
RUN mkdir -p ~/.vnc && \
    echo "octane" | vncpasswd -f > ~/.vnc/passwd && \
    chmod 600 ~/.vnc/passwd

# Create VNC startup script
RUN mkdir -p /root/.vnc && \
    echo '#!/bin/bash' > /root/.vnc/xstartup && \
    echo 'unset SESSION_MANAGER' >> /root/.vnc/xstartup && \
    echo 'unset DBUS_SESSION_BUS_ADDRESS' >> /root/.vnc/xstartup && \
    echo 'exec startxfce4' >> /root/.vnc/xstartup && \
    chmod +x /root/.vnc/xstartup

# Create Desktop directory and helper scripts
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

# ── Depth Anything V3 (DA3) dependencies ─────────────────────────────────
# These are needed for the depth_estimation_node (monocular metric depth).
#
# On x86 (desktop/laptop with RTX GPU):
#   Standard pip wheels work. torch auto-detects CUDA.
#
# On aarch64 (Jetson AGX Orin / JetPack):
#   torch, torchvision, and xformers MUST come from NVIDIA's JetPack-compatible
#   wheels, NOT regular pip. Replace the pip install below with:
#     pip install --no-cache-dir \
#       torch torchvision     # ← from NVIDIA's PyPI index for JetPack
#       xformers              # ← build from source or use dusty-nv container
#   See: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/
#
# The depth_anything_3 package itself is pure Python and works on both archs.
RUN pip install --no-cache-dir \
    torch torchvision \
    xformers \
    einops \
    huggingface_hub \
    safetensors \
    omegaconf \
    trimesh \
    open3d \
    imageio \
    "numpy<2" \
    && pip install --no-cache-dir git+https://github.com/ByteDance-Seed/Depth-Anything-3.git

WORKDIR /workspace

# Expose VNC port
EXPOSE 5901

