#!/bin/bash
# Veritas Engine setup script

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "=========================================="
echo "  Veritas Engine Setup"
echo "=========================================="
echo ""

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created at $VENV_DIR"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "Installing core dependencies..."
pip install \
    pydantic \
    pydantic-settings \
    pyyaml \
    python-dotenv \
    httpx \
    rich \
    typer \
    fire \
    fastapi \
    uvicorn

# Install optional dependencies (with fallback)
echo "Installing optional dependencies..."
pip install kuzu || echo "Warning: kuzu installation failed, using mock mode"
pip install lancedb || echo "Warning: lancedb installation failed, using mock mode"
pip install pandas numpy scipy scikit-learn || echo "Warning: data science packages failed"

# Install project
echo "Installing Veritas Engine..."
cd "$PROJECT_DIR"
pip install -e . --no-deps

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Quick start:"
echo "  $PROJECT_DIR/scripts/start.sh info"
echo "  $PROJECT_DIR/scripts/start.sh status"
echo "  $PROJECT_DIR/scripts/start.sh run '优化产线温度控制'"
echo ""
