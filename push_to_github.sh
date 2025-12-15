#!/bin/bash
# Script to push code to a new GitHub repository

set -e

echo "🚀 GitHub Repository Push Script"
echo "=================================="
echo ""

# Check if repository URL is provided
if [ -z "$1" ]; then
    echo "❌ Error: Repository URL not provided"
    echo ""
    echo "Usage: ./push_to_github.sh https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
    echo ""
    echo "📋 Steps:"
    echo "1. Go to https://github.com/new"
    echo "2. Create a new repository (DO NOT initialize with README)"
    echo "3. Copy the repository URL"
    echo "4. Run: ./push_to_github.sh <REPO_URL>"
    echo ""
    exit 1
fi

REPO_URL="$1"

echo "📦 Repository URL: $REPO_URL"
echo ""

# Remove old remote if exists
if git remote get-url origin &>/dev/null; then
    echo "🔄 Removing old remote 'origin'..."
    git remote remove origin
fi

# Add new remote
echo "➕ Adding new remote 'origin'..."
git remote add origin "$REPO_URL"

# Show current status
echo ""
echo "📊 Current git status:"
git status --short

echo ""
read -p "🤔 Do you want to push to GitHub now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Pushing to GitHub..."
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Successfully pushed to GitHub!"
        echo "🌐 View your repo at: $REPO_URL"
    else
        echo ""
        echo "❌ Push failed. You may need to:"
        echo "   1. Authenticate with GitHub (use Personal Access Token)"
        echo "   2. Or set up SSH keys"
        echo ""
        echo "For HTTPS authentication, use:"
        echo "   git config --global credential.helper store"
        echo "   (Then enter your GitHub username and Personal Access Token when prompted)"
    fi
else
    echo "⏸️  Push cancelled. Remote 'origin' has been set."
    echo "   Run 'git push -u origin main' when ready."
fi












