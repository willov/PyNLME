#!/bin/bash
# Version bumping script for PyNLME
# Usage: ./scripts/bump-version.sh [major|minor|patch|version]

set -e

SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

function usage() {
    echo "Usage: $0 [major|minor|patch|X.Y.Z]"
    echo ""
    echo "Examples:"
    echo "  $0 patch     # 0.1.0 -> 0.1.1"
    echo "  $0 minor     # 0.1.0 -> 0.2.0"
    echo "  $0 major     # 0.1.0 -> 1.0.0"
    echo "  $0 1.2.3     # Set version to 1.2.3"
    echo ""
    echo "This will:"
    echo "  1. Update version in pyproject.toml"
    echo "  2. Run tests to ensure everything works"
    echo "  3. Show next steps for release"
    exit 1
}

function get_current_version() {
    grep '^version = ' "$PROJECT_ROOT/pyproject.toml" | sed 's/version = "\(.*\)"/\1/'
}

function set_version() {
    local new_version="$1"
    
    # Validate version format
    if [[ ! "$new_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
        echo "❌ Invalid version format: $new_version"
        echo "Expected format: X.Y.Z or X.Y.Z-suffix (e.g., 1.0.0, 1.0.0-alpha)"
        exit 1
    fi
    
    echo "📝 Updating version to $new_version in pyproject.toml..."
    sed -i "s/^version = .*/version = \"$new_version\"/" "$PROJECT_ROOT/pyproject.toml"
}

function calculate_next_version() {
    local current="$1"
    local bump_type="$2"
    
    # Split version into components
    IFS='.' read -r major minor patch <<< "$current"
    
    case "$bump_type" in
        major)
            echo "$((major + 1)).0.0"
            ;;
        minor)
            echo "$major.$((minor + 1)).0"
            ;;
        patch)
            echo "$major.$minor.$((patch + 1))"
            ;;
        *)
            echo "❌ Unknown bump type: $bump_type"
            exit 1
            ;;
    esac
}

# Check arguments
if [ $# -ne 1 ]; then
    usage
fi

BUMP_TYPE="$1"

# Get current version
CURRENT_VERSION=$(get_current_version)
echo "📦 Current version: $CURRENT_VERSION"

# Calculate new version
if [[ "$BUMP_TYPE" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
    # Direct version specified
    NEW_VERSION="$BUMP_TYPE"
elif [[ "$BUMP_TYPE" =~ ^(major|minor|patch)$ ]]; then
    # Bump type specified
    NEW_VERSION=$(calculate_next_version "$CURRENT_VERSION" "$BUMP_TYPE")
else
    echo "❌ Invalid argument: $BUMP_TYPE"
    usage
fi

echo "🎯 New version: $NEW_VERSION"

# Check if version already exists as a tag
if git rev-parse "v$NEW_VERSION" >/dev/null 2>&1; then
    echo "⚠️  Warning: Tag v$NEW_VERSION already exists!"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Update version
set_version "$NEW_VERSION"

echo "✅ Version updated successfully!"

# Run basic validation
echo ""
echo "🧪 Running quick validation..."
cd "$PROJECT_ROOT"

# Check if uv is available
if command -v uv >/dev/null 2>&1; then
    # Quick test to ensure package still works
    echo "Running quick tests..."
    uv run pytest tests/test_basic.py -v -q || {
        echo "❌ Tests failed! Reverting version change..."
        set_version "$CURRENT_VERSION"
        exit 1
    }
    echo "✅ Quick tests passed"
else
    echo "⚠️  UV not found, skipping tests"
fi

echo ""
echo "🎉 Version bump complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Review the changes:"
echo "     git diff pyproject.toml"
echo ""
echo "  2. Update CHANGELOG.md with release notes for v$NEW_VERSION"
echo ""
echo "  3. For automatic release:"
echo "     git add pyproject.toml CHANGELOG.md"
echo "     git commit -m 'Prepare release v$NEW_VERSION'"
echo "     git push origin main"
echo "     # GitHub Actions will automatically create the release"
echo ""
echo "  4. For manual release:"
echo "     git add pyproject.toml CHANGELOG.md"
echo "     git commit -m 'Bump version to v$NEW_VERSION'"
echo "     git tag -a v$NEW_VERSION -m 'Release v$NEW_VERSION'"
echo "     git push origin main --tags"
echo ""
echo "🔗 Monitor release progress at:"
echo "   https://github.com/willov/PyNLME/actions"
