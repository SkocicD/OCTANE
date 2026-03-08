FROM ros:humble

SHELL ["/bin/bash", "-c"]

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# ROS 2 bridge defaults — overridden at runtime via docker-compose / .env
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV ROS_DOMAIN_ID=0

# Install ROS dependencies
RUN apt update && apt install -y \
    python3-colcon-common-extensions \
    ros-humble-backward-ros \
    ros-humble-cv-bridge \
    ros-humble-diagnostic-updater \
    ros-humble-image-transport \
    ros-humble-rclcpp-components \
    ros-humble-rmw-fastrtps-cpp \
    ros-humble-tf2-sensor-msgs \
    libssl-dev \
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

WORKDIR /workspace

# Expose VNC port
EXPOSE 5901

