#!/bin/bash

# Safe push script for Photosight
# Implements the safe-push workflow with Claude Code Review integration

set -e

echo "🚀 Starting Photosight safe push workflow..."

# Function to check if we're in a git repository
check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        echo "❌ Not in a git repository"
        exit 1
    fi
}

# Function to check for staged changes
check_staged_changes() {
    if ! git diff --cached --quiet; then
        return 0  # Has staged changes
    else
        echo "❌ No staged changes found. Please stage your changes first with: git add <files>"
        exit 1
    fi
}

# Function to validate branch name
validate_branch_name() {
    current_branch=$(git branch --show-current)
    echo "Current branch: $current_branch"
    
    if [[ ! "$current_branch" =~ ^feature/issue-[0-9]+-.*$ ]] && [[ "$current_branch" != "main" ]] && [[ "$current_branch" != "master" ]]; then
        echo "⚠️ Branch name doesn't match recommended pattern: feature/issue-XX-description"
        echo "Do you want to create a properly named branch? (y/n)"
        read -r create_branch
        
        if [[ $create_branch == "y" || $create_branch == "Y" ]]; then
            echo "Enter issue number:"
            read -r issue_id
            echo "Enter brief description (use hyphens, no spaces):"
            read -r description
            
            new_branch="feature/issue-${issue_id}-${description}"
            echo "Creating branch: $new_branch"
            git checkout -b "$new_branch"
        fi
    fi
}

# Function to perform privacy and security checks
privacy_security_check() {
    echo "🔍 Performing privacy and security checks..."
    
    # Check for dangerous file types
    dangerous_files=$(git status --porcelain | grep -E '\.(jpg|jpeg|png|gif|bmp|tiff|raw|heic|pkl|pt|pth|onnx|h5|pb|weights)$' || true)
    if [ ! -z "$dangerous_files" ]; then
        echo "🚨 DANGER: Found sensitive files:"
        echo "$dangerous_files"
        echo "Unstaging dangerous files..."
        git reset HEAD *.jpg *.jpeg *.png *.gif *.pkl *.pt *.onnx 2>/dev/null || true
        echo "❌ Please review and manually stage only safe files"
        exit 1
    fi
    
    # Search for privacy-sensitive patterns in staged files
    privacy_patterns=$(git diff --cached | grep -i -E 'personal|private|test.*image|sample.*photo|face.*recognition|identity|facial.*id' || true)
    if [ ! -z "$privacy_patterns" ]; then
        echo "⚠️ WARNING: Potential privacy-sensitive content found:"
        echo "$privacy_patterns" | head -10
        echo "Do you want to continue? (y/n)"
        read -r continue_anyway
        
        if [[ $continue_anyway != "y" && $continue_anyway != "Y" ]]; then
            echo "❌ Aborting push. Please review and remove privacy-sensitive content."
            exit 1
        fi
    fi
    
    # Check for embedded image data
    embedded_data=$(git diff --cached -- "*.py" "*.json" "*.js" | grep -i -E 'image.*data|base64.*[A-Za-z0-9+/]{50,}|blob.*data' || true)
    if [ ! -z "$embedded_data" ]; then
        echo "⚠️ Found potential embedded image data - review carefully"
        echo "Do you want to continue? (y/n)"
        read -r continue_embedded
        
        if [[ $continue_embedded != "y" && $continue_embedded != "Y" ]]; then
            echo "❌ Aborting push. Please review embedded data."
            exit 1
        fi
    fi
    
    echo "✅ Privacy and security checks passed"
}

# Function to generate commit message
generate_commit_message() {
    echo "📝 Generating commit message..."
    
    # Analyze changed files to determine commit type
    if git diff --cached --name-only | grep -q "photosight/ai/"; then
        commit_type="feat"
        commit_scope="ai"
    elif git diff --cached --name-only | grep -q "photosight/processing/"; then
        commit_type="feat"
        commit_scope="processing"
    elif git diff --cached --name-only | grep -q "tests/"; then
        commit_type="test"
        commit_scope="tests"
    elif git diff --cached --name-only | grep -q "docs/"; then
        commit_type="docs"
        commit_scope="docs"
    elif git diff --cached --name-only | grep -q "scripts/"; then
        commit_type="chore"
        commit_scope="scripts"
    else
        commit_type="fix"
        commit_scope="general"
    fi
    
    echo "Enter commit description:"
    read -r commit_description
    
    commit_msg="${commit_type}(${commit_scope}): ${commit_description}"
    echo "Commit message: $commit_msg"
    
    # Commit changes
    git commit -m "$commit_msg"
}

# Function to push and create PR
push_and_create_pr() {
    current_branch=$(git branch --show-current)
    
    # Check if branch is main/master
    if [[ "$current_branch" == "main" || "$current_branch" == "master" ]]; then
        echo "🚨 ERROR: Cannot push directly to main/master branch!"
        echo "Please create a feature branch first."
        exit 1
    fi
    
    echo "🚀 Pushing to origin..."
    git push -u origin "$current_branch"
    
    # Check if gh CLI is available
    if ! command -v gh &> /dev/null; then
        echo "⚠️ GitHub CLI not found. Please install with: brew install gh"
        echo "Your branch has been pushed. Create PR manually at GitHub."
        return
    fi
    
    # Create PR
    echo "📝 Creating pull request..."
    echo "Enter PR title (or press enter for default):"
    read -r pr_title
    
    if [ -z "$pr_title" ]; then
        pr_title="[$current_branch] Photosight feature implementation"
    fi
    
    pr_body="## Summary
- Computer vision feature implementation
- Privacy-preserving design
- No personal images or test photos included

## Test Plan
- [ ] Verified no personal images in changes
- [ ] Tested with synthetic/public domain images
- [ ] Validated crop suggestions are privacy-safe
- [ ] Checked model file handling

## Privacy Compliance
- [ ] No facial recognition of real people
- [ ] No personal photo data embedded
- [ ] Crop suggestions preserve privacy

🤖 Automated Claude Code Review will run on this PR"
    
    pr_url=$(gh pr create \
        --title "$pr_title" \
        --body "$pr_body" \
        --base main \
        --head "$current_branch")
    
    echo "✅ PR created: $pr_url"
    echo "🤖 Claude Code Review will run automatically"
    
    # Open PR in browser (optional)
    echo "Open PR in browser? (y/n)"
    read -r open_browser
    if [[ $open_browser == "y" || $open_browser == "Y" ]]; then
        gh pr view --web
    fi
}

# Main workflow
main() {
    check_git_repo
    check_staged_changes
    validate_branch_name
    privacy_security_check
    generate_commit_message
    push_and_create_pr
    
    echo "🎉 Safe push completed successfully!"
    echo "📋 Next steps:"
    echo "1. Monitor the PR for Claude Code Review results"
    echo "2. Address any review feedback"
    echo "3. Merge when approved"
}

# Run main function
main "$@"