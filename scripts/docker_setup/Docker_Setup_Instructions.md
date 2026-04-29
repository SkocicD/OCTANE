# DOCKER SETUP INSTRUCTIONS

## Current Setup

Your Docker environment is now configured with a layered approach using NVIDIA's Isaac ROS base image.

## Directory Structure

```
/home/csulunabotics/OCTANE/scripts/docker_setup/
├── Dockerfile              # Main application Dockerfile
├── Dockerfile.layers       # Multi-stage Dockerfile for layered builds
├── Dockerfile.multistage   # Alternative multi-stage Dockerfile with build args
├── Dockerfile.dev           # Development environment Dockerfile
├── Docker_Guide.md         # Quick reference guide
├── setup_isaac_docker.sh   # Complete setup script
├── build_docker.sh         # Flexible build script with dependency/app layers
├── build_layered.sh        # Layered build script
├── clear_docker_images.sh   # Cleanup script (preserves base image)
├── manage_base_image.sh   # Base image management
├── auth_ngc.sh            # NVIDIA authentication
└── setup_docker_storage.sh # Storage configuration
```

## How the Layered Approach Works

### Three-Layer Docker System

1. **Base Layer**: NVIDIA Isaac ROS Base Image
   - Pre-built image: `nvcr.io/nvidia/isaac/ros:isaac_ros_054e16b5c3a328b621af47d26009c348-arm64-fastos`
   - Contains ROS2, CUDA, TensorRT, and NVIDIA packages
   - Downloaded once and preserved

2. **Dependencies Layer**: System dependencies and packages (infrequently changed)
   - ROS packages, system libraries, Python dependencies
   - Built once and reused

3. **Application Layer**: Your Custom Code (frequently changed)
   - Your source code and configurations
   - Rebuilds quickly when you make changes

## Setup Process

### Flexible Build System
```bash
# Build only dependencies layer (when adding new packages)
./build_docker.sh --deps

# Build only application layer (when changing code)
./build_docker.sh --app

# Build both layers (default)
./build_docker.sh

# Clean previous builds to prevent clutter
./build_docker.sh --clean

# Examples:
./build_docker.sh --deps --clean    # Rebuild only dependencies
./build_docker.sh --app             # Rebuild only application code
./build_docker.sh                   # Rebuild everything
```

## Key Benefits

1. **Base Image Preservation**: The expensive NVIDIA base image is downloaded once and never deleted
2. **Fast Build Times**: Only your application layer rebuilds
3. **Proper Storage**: Docker data stored on NVMe drive (445GB free)
4. **Automatic Cleanup**: Cleanup scripts preserve base image while removing temporary data
5. **Flexible Builds**: Build only what you need (dependencies or application)

## Important Scripts

- `setup_isaac_docker.sh`: Complete initial setup
- `build_docker.sh`: Flexible build system (--deps, --app, --clean)
- `manage_base_image.sh`: Preserve base image
- `clear_docker_images.sh`: Safe cleanup
- `auth_ngc.sh`: Handle NVIDIA authentication

The system is designed so that the expensive base image (with CUDA/TensorRT) is downloaded once and preserved, while only your application code rebuilds when you make changes. This gives you much faster iteration times.