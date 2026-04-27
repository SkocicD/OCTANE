#!/bin/bash

# Script to authenticate with NVIDIA NGC for Isaac ROS container access
echo "Setting up NVIDIA NGC authentication..."

# Check if NGC API key file exists
if [ ! -f "/home/csulunabotics/ngc-api-key.txt" ]; then
    echo "Error: NGC API key file not found at /home/csulunabotics/ngc-api-key.txt"
    echo "Please create this file with your NGC API key."
    exit 1
fi

# Read the API key
NGC_API_KEY=$(cat /home/csulunabotics/ngc-api-key.txt)

# Authenticate with Docker
echo "Authenticating with NVIDIA NGC..."
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

if [ $? -eq 0 ]; then
    echo "Successfully authenticated with NVIDIA NGC!"
    echo "You can now pull Isaac ROS images from nvcr.io"
else
    echo "Failed to authenticate with NVIDIA NGC."
    echo "Please check your API key and try again."
    exit 1
fi