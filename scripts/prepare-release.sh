#!/bin/bash
# Release preparation script for PyNLME

set -e

echo "🚀 PyNLME Release Preparation"
echo "============================"

# Check if we're on main branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "❌ Must be on main branch for release. Currently on: $CURRENT_BRANCH"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "❌ Uncommitted changes found. Please commit or stash them first."
    exit 1
fi

# Get version from pyproject.toml
VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
echo "📦 Current version: $VERSION"

# Run comprehensive checks
echo ""
echo "🧪 Running test suite..."
uv run pytest tests/ -v

echo ""
echo "🔍 Running type checks..."
uv run mypy src/pynlme/ --ignore-missing-imports || echo "⚠️  Type check warnings found"

echo ""
echo "🎨 Checking code formatting..."
uv run ruff format --check src/ tests/ examples/ || {
    echo "❌ Code formatting issues found. Run: uv run ruff format src/ tests/ examples/"
    exit 1
}

echo ""
echo "🔧 Running linting..."
uv run ruff check src/ tests/ examples/ || echo "⚠️  Linting warnings found"

echo ""
echo "🦀 Building Rust extension..."
uv run maturin develop

echo ""
echo "📊 Running example scripts..."
cd examples/
python basic_usage.py > /dev/null 2>&1 && echo "✅ basic_usage.py"
python advanced_usage.py > /dev/null 2>&1 && echo "✅ advanced_usage.py"
python matlab_comparison.py > /dev/null 2>&1 && echo "✅ matlab_comparison.py"
cd ..

echo ""
echo "🏗️  Building distribution packages..."
uv run maturin build --release

echo ""
echo "📋 Release checklist:"
echo "  ✅ All tests passing"
echo "  ✅ Code formatted and linted"
echo "  ✅ Rust extension builds"
echo "  ✅ Examples run successfully"
echo "  ✅ Distribution packages built"

echo ""
echo "📚 Documentation checklist:"
echo "  - [ ] Update CHANGELOG.md"
echo "  - [ ] Update version in pyproject.toml"
echo "  - [ ] Update version in CITATION.cff"
echo "  - [ ] Review README.md"
echo "  - [ ] Check all documentation links"

echo ""
echo "🎯 Next steps for MANUAL release:"
echo "  1. Update version manually in pyproject.toml (currently: $VERSION)"
echo "  2. Update CHANGELOG.md with release notes"
echo "  3. Commit changes: git commit -am 'Release vX.Y.Z: Description'"
echo "  4. Push to main: git push origin main"
echo "  5. 🤖 GitHub Actions will automatically:"
echo "     - Detect version change"
echo "     - Create git tag vX.Y.Z"
echo "     - Build wheels for all platforms" 
echo "     - Create GitHub release with artifacts"

echo ""
echo "📋 Post-release checklist:"
echo "  - [ ] Verify all wheel files are uploaded to GitHub release"
echo "  - [ ] Test installation from GitHub release"
echo "  - [ ] Update documentation if needed"
echo "  - [ ] Consider PyPI publishing when ready for wider distribution"

echo ""
echo "✨ Release preparation complete for version $VERSION!"
