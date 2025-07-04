#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to display error messages and exit
error_exit() {
  echo "$1" >&2
  echo "$(date): $1" >> error.log  # Log error to a file for debugging
  exit 1
}

# 1. Set ENV_NAME with default value "phishpedia" if not already set
ENV_NAME="${ENV_NAME:-phishpedia}"

# 2. Check if ENV_NAME is set (it always will be now, but kept for flexibility)
if [ -z "$ENV_NAME" ]; then
  error_exit "ENV_NAME is not set. Please set the environment name and try again."
fi

# 3. Set retry count for downloads
RETRY_COUNT=3

# 4. Function to download files with retry mechanism
download_with_retry() {
  local file_id="$1"
  local file_name="$2"
  local count=0

  until [ $count -ge $RETRY_COUNT ]
  do
    echo "Attempting to download $file_name (Attempt $((count + 1))/$RETRY_COUNT)..."
    conda run -n "$ENV_NAME" gdown --id "$file_id" -O "$file_name" && break
    count=$((count + 1))
    echo "Retry $count of $RETRY_COUNT for $file_name..."
    sleep 2  # Increased wait time to 2 seconds
  done

  if [ $count -ge $RETRY_COUNT ]; then
    error_exit "Failed to download $file_name after $RETRY_COUNT attempts."
  fi
}

# 5. Ensure Conda is installed
if ! command -v conda &> /dev/null; then
  error_exit "Conda is not installed. Please install Conda and try again."
fi

# 6. Initialize Conda for bash
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 7. Check if the environment exists
if conda info --envs | grep -w "^$ENV_NAME" > /dev/null 2>&1; then
  echo "Activating existing Conda environment: $ENV_NAME"
else
  echo "Creating new Conda environment: $ENV_NAME with Python 3.10"
  conda create -y -n "$ENV_NAME" python=3.10
fi

# activate the Conda environment
echo "Activating Conda environment: $ENV_NAME"
conda activate "$ENV_NAME"

# 10. Determine the Operating System
OS=$(uname -s)

# Install dependencies
conda run -n "$ENV_NAME" pip3 install torch torchvision torchaudio
conda run -n "$ENV_NAME" pip install gdown opencv-python numpy Pillow matplotlib

mkdir models/
cd models/
download_with_retry "1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS" "resnetv2_rgb_new.pth.tar"

download_with_retry "1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I" "expand_targetlist.zip"
unzip expand_targetlist.zip -d expand_targetlist
# Change to the extracted directory
cd expand_targetlist || exit 1  # Exit if the directory doesn't exist

# Check if there's a nested 'expand_targetlist/' directory
if [ -d "expand_targetlist" ]; then
  echo "Nested directory 'expand_targetlist/' detected. Moving contents up..."

  # Enable dotglob to include hidden files
  shopt -s dotglob

  # Move everything from the nested directory to the current directory
  mv expand_targetlist/* .

  # Disable dotglob to revert back to normal behavior
  shopt -u dotglob

  # Remove the now-empty nested directory
  rmdir expand_targetlist
  cd ../
else
  echo "No nested 'expand_targetlist/' directory found. No action needed."
fi

echo "Extraction completed successfully."
echo "All packages installed and models downloaded successfully!"


