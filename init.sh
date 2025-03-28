#!/bin/bash

# Define Miniconda version and installation directory
MINICONDA_VERSION="latest"
INSTALL_DIR="$HOME/miniconda"

# Determine OS and architecture
OS="Linux"  # Default to Linux
ARCH="aarch64"  # Set for ARM64/aarch64

if [ "$(uname)" == "Darwin" ]; then
    OS="MacOSX"
fi

# Construct Miniconda download URL
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-$OS-$ARCH.sh"

# Download Miniconda installer
echo "Downloading Miniconda from: $MINICONDA_URL"
curl -o miniconda.sh -L $MINICONDA_URL

# Install Miniconda
echo "Installing Miniconda to: $INSTALL_DIR"
bash miniconda.sh -b -p $INSTALL_DIR

# Clean up installer
rm miniconda.sh

# Initialize conda
echo "Initializing conda"
source "$INSTALL_DIR/bin/activate"
conda init

echo "Miniconda installation complete. Please restart your terminal or run 'source ~/.bashrc' to activate conda."

# Create and activate the environment
source ~/.bashrc
conda create --name ai4gcc python=3.10 -y
conda activate ai4gcc
pip install -r requirements_simple.txt
echo "ai4gcc env preparing complete. Run 'conda activate ai4gcc' to use it."