#!/usr/bin/env bash
# AI File Discovery — Quick Install Script
# Usage: curl -fsSL https://raw.githubusercontent.com/girdav01/AIFileDisco/main/install.sh | bash
set -euo pipefail

REPO="https://github.com/girdav01/AIFileDisco.git"
INSTALL_DIR="${HOME}/.aifilefinder"

echo ""
echo "  ╔══════════════════════════════════╗"
echo "  ║   AI File Discovery — Install    ║"
echo "  ╚══════════════════════════════════╝"
echo ""

# Check Python 3.8+
if ! command -v python3 &>/dev/null; then
    echo "  ✗ Python 3 is required but not found."
    echo "    Install it from https://python.org"
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 8 ]); then
    echo "  ✗ Python 3.8+ required (found $PY_VERSION)"
    exit 1
fi
echo "  ✓ Python $PY_VERSION"

# Clone or update
if [ -d "$INSTALL_DIR" ]; then
    echo "  ↻ Updating existing installation..."
    cd "$INSTALL_DIR"
    git pull --quiet
else
    echo "  ↓ Cloning repository..."
    git clone --quiet "$REPO" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Install
echo "  ⚙ Installing..."
pip install --quiet . 2>/dev/null || pip install --quiet --user .

# Verify
if command -v aifiles &>/dev/null; then
    echo "  ✓ CLI installed: aifiles"
else
    echo "  ⚠ 'aifiles' not in PATH — you may need to add ~/.local/bin to your PATH"
fi

if command -v aifiles-server &>/dev/null; then
    echo "  ✓ Server installed: aifiles-server"
fi

echo ""
echo "  ✅ Installation complete!"
echo ""
echo "  Usage:"
echo "    aifiles /path/to/scan          # CLI scan"
echo "    aifiles /path --integrity      # Integrity check"
echo "    aifiles-server                  # Web dashboard at http://localhost:8505"
echo ""
