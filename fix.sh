#!/bin/bash

# This script initializes a Python virtual environment, updates .gitignore,
# and generates a requirements.txt file in the current directory.

# Exit on error
set -e

# Directory containing this script (assumed to be the transformer directory)
DIR=$(pwd)

# Step 1: Create and activate a virtual environment
echo "Creating and activating virtual environment in $DIR..."
if [ ! -d "$DIR/venv" ]; then
    python3 -m venv "$DIR/venv"
else
    echo "Virtual environment already exists at $DIR/venv"
fi
source "$DIR/venv/bin/activate"

# Step 2: Install required packages
echo "Installing required packages..."
pip install torch numpy

# Step 3: Update requirements.txt
echo "Generating requirements.txt..."
pip freeze > "$DIR/requirements.txt"
echo "Updated requirements.txt with current dependencies"

# Step 4: Update .gitignore
echo "Updating .gitignore..."
cat << EOF > "$DIR/.gitignore"
# Python virtual environment
venv/

# Python cache files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
dist/
*.egg-info/

# macOS
.DS_Store

# IDEs and editors
.vscode/
.idea/

# Jupyter Notebook
.ipynb_checkpoints/

# Logs and temporary files
*.log
*.tmp
EOF
echo "Updated .gitignore with common Python ignores"

# Step 5: Display completion message
echo "Setup complete! Virtual environment activated, requirements.txt updated, and .gitignore updated."

# Keep the virtual environment activated for the user
echo "You are now in the virtual environment. To deactivate, run 'deactivate'."