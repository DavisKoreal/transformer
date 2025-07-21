#!/bin/bash

# This script applies changes to the transformer repository for XAUUSD time series forecasting.
# It updates transformer.py, .gitignore, requirements.txt, and adds a README.md.

# Exit on error
set -e

# Directory containing this script
DIR=$(pwd)

# Step 1: Ensure virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Virtual environment not activated. Activating $DIR/venv..."
    if [ ! -d "$DIR/venv" ]; then
        echo "Error: Virtual environment not found at $DIR/venv"
        exit 1
    fi
    source "$DIR/venv/bin/activate"
else
    echo "Virtual environment already activated: $VIRTUAL_ENV"
fi

# Step 2: Install required packages
echo "Installing required packages..."
pip install torch numpy pandas scikit-learn ta

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

# Data files
*.csv
*.parquet
*.h5
