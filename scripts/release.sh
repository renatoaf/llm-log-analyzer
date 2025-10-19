#!/bin/bash

# Release script for llm-log-analyzer
# Usage: ./scripts/release.sh [version] [--test]
# Example: ./scripts/release.sh 1.0.1
# Example: ./scripts/release.sh 1.0.0-beta1 --test

set -e

VERSION=$1

TAG="$VERSION"

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 1.0.1"
    exit 1
fi

echo "🚀 Preparing release $VERSION"

# Update version in pyproject.toml
echo "📝 Updating version in pyproject.toml..."
sed -i.bak "s/version = \".*\"/version = \"$VERSION\"/" pyproject.toml
rm pyproject.toml.bak

# Check if there are uncommitted changes
if ! git diff --quiet; then
    echo "📋 Committing version update..."
    git add pyproject.toml
    git commit -m "Bump version to $VERSION"
fi

# Build and test locally
echo "🔨 Building package..."
rm -rf dist/ build/
python -m build

echo "✅ Checking package..."
twine check dist/*

# Create and push tag
echo "📤 Creating and pushing tag $TAG..."
git tag "$TAG"
git push origin main
git push origin "$TAG"

echo "✨ Release $VERSION initiated!"
echo "🔗 Monitor the GitHub Actions at: https://github.com/renatoaf/llm-log-analyzer/actions"
