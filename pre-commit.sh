#!/bin/bash
# Pre-commit checks — run before committing

echo "Checking for secrets..."
if grep -rE "AIza[0-9A-Za-z-_]{35}|sk-[0-9a-zA-Z]{32,}" src/ tests/ 2>/dev/null; then
    echo "Potential secrets detected — aborting"
    exit 1
fi

echo "Running tests..."
uv run python -m pytest tests/ -v
if [ $? -ne 0 ]; then exit 1; fi

echo "All checks passed."
