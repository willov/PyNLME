#!/bin/bash
# Build wheels locally for testing

set -e

echo "🔧 Building PyNLME wheels locally"
echo "================================="

# Install cibuildwheel if not present
pip install cibuildwheel

# Build wheels
echo "🏗️  Building wheels..."
cibuildwheel --output-dir wheelhouse

echo "📦 Built wheels:"
ls -la wheelhouse/

echo ""
echo "✅ Wheel building complete!"
echo ""
echo "To test a wheel:"
echo "  pip install wheelhouse/pynlme-*.whl"
echo ""
echo "For official releases:"
echo "  - Create a git tag: git tag v0.1.0 && git push --tags"
echo "  - GitHub Actions will build and upload wheels automatically"
echo "  - Wheels will be available at: https://github.com/willov/PyNLME/releases"
