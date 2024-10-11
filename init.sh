#!/bin/bash

# Define Miniconda version and installation directory
MINICONDA_VERSION="latest"
INSTALL_DIR="$HOME/miniconda"

# Determine OS and architecture
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="MacOSX"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Construct Miniconda download URL
if [[ $(uname -m) == "x86_64" ]]; then
    ARCH="x86_64"
else
    echo "Unsupported architecture: $(uname -m)"
    exit 1
fi

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
