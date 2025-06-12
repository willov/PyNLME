#!/bin/bash
# Build wheels locally for testing

set -e

echo "🔧 Building PyNLME wheels locally"
echo "================================="

# Set up the project
echo "📦 Setting up project dependencies..."
uv sync

# Build wheels
echo "🏗️  Building wheels..."
uv run cibuildwheel --output-dir wheelhouse

echo "📦 Built wheels:"
ls -la wheelhouse/

echo ""
echo "✅ Wheel building complete!"
echo ""
echo "To test a wheel:"
echo "  uv pip install wheelhouse/pynlme-*.whl"
echo ""
echo "For official releases:"
echo "  - Create a git tag: git tag v0.1.0 && git push --tags"
echo "  - GitHub Actions will build and upload wheels automatically"
echo "  - Wheels will be available at: https://github.com/willov/PyNLME/releases"
