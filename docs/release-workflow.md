# PyNLME Release Workflow

This document shows the **manual version control** workflow for PyNLME releases.

## 🎯 Release Process Overview

PyNLME uses **manual version setting** combined with **automatic release creation**:

1. **Developer manually sets version** in `pyproject.toml`
2. **GitHub Actions detects change** and creates release automatically
3. **Users get pre-built wheels** without needing Rust

## 📋 Step-by-Step Release

### 1. Prepare for Release

```bash
# Run validation script
./scripts/prepare-release.sh

# This checks:
# - All tests pass
# - Code is formatted
# - Rust builds correctly
# - Examples work
```

### 2. Update Version Manually

Edit `pyproject.toml`:

```toml
[project]
name = "pynlme"
version = "0.1.1"  # <- Change this manually (was 0.1.0)
```

### 3. Update Changelog

Edit `CHANGELOG.md`:

```markdown
## [0.1.1] - 2025-06-11

### Added
- New feature X
- Enhanced Y functionality

### Fixed
- Bug Z in algorithm
- Documentation updates
```

### 4. Commit and Push

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "Release v0.1.1: Add feature X, fix bug Z"
git push origin main
```

### 5. GitHub Actions Magic ✨

GitHub Actions automatically:
- **Detects** version change (`0.1.0` → `0.1.1`)
- **Validates** that tag `v0.1.1` doesn't exist
- **Runs tests** to ensure everything works
- **Creates git tag** `v0.1.1`
- **Builds wheels** for Linux, Windows, macOS
- **Creates GitHub release** with all wheels attached
- **Marks as pre-release** (proof-of-concept status)

### 6. Users Install Easily

Users can now install without Rust:

```bash
# Install latest wheel from GitHub releases
pip install https://github.com/willov/PyNLME/releases/latest/download/pynlme-0.1.1-cp38-abi3-linux_x86_64.whl

# Or browse releases and pick specific wheel
# https://github.com/willov/PyNLME/releases
```

## 🔍 What GitHub Actions Checks

The auto-release workflow (`auto-release.yml`) checks:

- ✅ **Version changed** in `pyproject.toml`
- ✅ **Tag doesn't exist** yet
- ✅ **Valid version format** (e.g., `1.2.3` or `1.2.3-alpha`)
- ✅ **Tests pass**
- ✅ **Code builds successfully**

## 🚫 When Releases Don't Happen

No release is created if:
- Version unchanged (e.g., just documentation updates)
- Tag already exists (prevents duplicate releases)
- Version format invalid (must be semantic versioning)
- Tests fail

## 💡 Benefits of This Approach

- ✅ **Manual control** - You decide when to release
- ✅ **Automatic execution** - No manual tag/release creation
- ✅ **Safe** - Won't create duplicate or broken releases
- ✅ **User-friendly** - Pre-built wheels, no Rust required
- ✅ **Clear history** - Git tags match release versions

## 🎯 Example Workflow

```bash
# Working on features...
git commit -m "Add new algorithm"
git commit -m "Fix edge case"
git commit -m "Update docs"

# Ready to release!
./scripts/prepare-release.sh  # Validate everything

# Edit pyproject.toml: version = "0.2.0"
# Edit CHANGELOG.md: ## [0.2.0] - 2025-06-11

git add pyproject.toml CHANGELOG.md
git commit -m "Release v0.2.0: Major algorithm improvements"
git push origin main

# 🤖 GitHub Actions automatically creates v0.2.0 release with wheels
# ✅ Users can install: pip install <wheel-url>
```

This workflow gives you full control over versioning while automating the tedious parts of release creation!
