FROM nvcr.io/nvidia/isaac/ros:isaac_ros_054e16b5c3a328b621af47d26009c348-arm64-fastos
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install essential packages
RUN apt update && apt install -y python3-colcon-common-extensions git
WORKDIR /workspace

# Install Orbbec SDK dependencies
RUN apt update && apt install -y libusb-1.0-0-dev

# Clone and build Orbbec SDK
RUN cd /workspace && \
    if [ ! -d "src/orbbec" ]; then \
    mkdir -p src && \
    cd src && \
    git clone https://github.com/orbbec/OrbbecSDK.git orbbec && \
    cd orbbec && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make -j4 && \
    make install; \
    fi

# Install comprehensive ROS dependencies for external packages
RUN apt update && apt install -y \
    ros-humble-desktop \
    ros-humble-camera-info-manager \
    ros-humble-cv-bridge \
    ros-humble-image-publisher \
    libuvc-dev \
    libgoogle-glog-dev \
    nlohmann-json3-dev \
    libeigen3-dev \
    libopenmpi-dev \
    python3-dev \
    cmake \
    build-essential

# Clone external packages (shallow clones for reliability)
RUN mkdir -p src/external_pkgs && \
    cd src/external_pkgs && \
    git clone --depth 1 https://github.com/orbbec/ros2_astra_camera.git && \
    git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git depth-anything-3

# Install Python dependencies for depth-anything-3 (excluding pycolmap - no aarch64 wheel)
RUN apt update && apt install -y \
    python3-pip \
    python3-venv \
    python3-numpy \
    python3-opencv \
    libglfw3-dev \
    libosmesa6-dev

# Install lightweight Python dependencies (skip torch/depth-anything-3 - install manually on Jetson)
RUN pip3 install --no-cache-dir \
    einops \
    huggingface_hub \
    imageio \
    "numpy<2" \
    opencv-python \
    fastapi \
    uvicorn \
    requests \
    typer \
    pillow \
    omegaconf \
    plyfile \
    safetensors \
    pillow_heif || \
    echo "Warning: Some Python packages failed to install"

# Note: PyTorch must be installed from NVIDIA JetPack for Jetson devices
# See: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson.html
# depth-anything-3 requires PyTorch + CUDA and will not be available in Docker build
# Run colcon build without depth-anything-3 for now

# Build ros2_astra_camera package
RUN cd /workspace && \
    colcon build --packages-select ros2_astra_camera --cmake-args -DCMAKE_BUILD_TYPE=Release || \
    echo "Warning: ros2_astra_camera build failed or not found"

# Set PYTHONPATH to include external packages
ENV PYTHONPATH=${PYTHONPATH}:/workspace/src/external_pkgs/depth-anything-3/src

# Source ROS2 setup by default
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
