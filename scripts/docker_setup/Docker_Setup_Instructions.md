# DOCKER SETUP INSTRUCTIONS

## Current Setup

Your Docker environment is now configured with a layered approach using NVIDIA's Isaac ROS base image.

## Directory Structure

```
/media/csulunabotics/SSD/OCTANE/scripts/docker_setup/
├── Dockerfile              # Main application Dockerfile
├── Docker_Guide.md         # Quick reference guide
├── setup_isaac_docker.sh   # Complete setup script
├── build_layered.sh        # Build your application layer
├── clear_docker_images.sh   # Cleanup script (preserves base image)
├── manage_base_image.sh   # Base image management
├── auth_ngc.sh            # NVIDIA authentication
└── setup_docker_storage.sh # Storage configuration
```

## How the Layered Approach Works

### Two-Layer Docker System

1. **Base Layer**: NVIDIA Isaac ROS Base Image
   - Pre-built image: `nvcr.io/nvidia/isaac/ros:isaac_ros_054e16b5c3a328b621af47d26009c348-arm64-fastos`
   - Contains ROS2, CUDA, TensorRT, and NVIDIA packages
   - Downloaded once and preserved

2. **Application Layer**: Your Custom Code
   - Everything in your Dockerfile after the `FROM` instruction
   - Rebuilds quickly when you make changes

## Setup Process

### Initial Setup (Run Once)
```bash
cd /media/csulunabotics/SSD/OCTANE/scripts/docker_setup/
./setup_isaac_docker.sh
```

This handles:
1. NVIDIA NGC authentication
2. Pulling and preserving the base image
3. Building your application layer

### Daily Usage
```bash
# Rebuild only your application layer (fast)
./build_layered.sh --layered

# Clean up temporary images (preserves base)
./clear_docker_images.sh
```

## Key Benefits

1. **Base Image Preservation**: The expensive NVIDIA base image is downloaded once and never deleted
2. **Fast Build Times**: Only your application layer rebuilds
3. **Proper Storage**: Docker data stored on NVMe drive (445GB free)
4. **Automatic Cleanup**: Cleanup scripts preserve base image while removing temporary data

## Important Scripts

- `setup_isaac_docker.sh`: Complete initial setup
- `build_layered.sh --layered`: Rebuild application layer
- `manage_base_image.sh`: Preserve base image
- `clear_docker_images.sh`: Safe cleanup
- `auth_ngc.sh`: Handle NVIDIA authentication

The system is designed so that the expensive base image (with CUDA/TensorRT) is downloaded once and preserved, while only your application code rebuilds when you make changes. This gives you much faster iteration times.