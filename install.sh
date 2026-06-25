#!/usr/bin/env bash
set -euo pipefail

# ArrowSpace Skills — install / register script
# Usage: bash install.sh [--opencode]

echo "=== ArrowSpace Skills Setup ==="

# 1. Ensure the core library is installed
if ! python -c "import arrowspace" 2>/dev/null; then
    echo "Installing arrowspace..."
    python -m pip install arrowspace
fi

# 2. Install this package
python -m pip install -e .

# 3. Register as opencode skill if requested
if [ "${1:-}" = "--opencode" ]; then
    OPENCODE_SKILLS="${HOME}/.config/opencode/skills"
    mkdir -p "${OPENCODE_SKILLS}/arrowspace"
    cp SKILL.md "${OPENCODE_SKILLS}/arrowspace/"
    echo "ArrowSpace skill registered in opencode."
fi

echo "Done. Try: python -c \"from arrowspace_skills import suggest_params; print(suggest_params(1000, 768))\""
