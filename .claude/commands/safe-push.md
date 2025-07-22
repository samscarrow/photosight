# Safe Push with PR Code Review for Photosight

Perform a safe git workflow with privacy checks, proper branch management, and automated PR code review for computer vision code.

## Usage
```
/safe-push
```

## What it does
1. Verifies current branch matches the feature pattern
2. Scans for personal images, model files, and sensitive data
3. Creates safe commit with proper message format
4. Pushes to feature branch
5. Creates PR with automated Claude Code Review
6. Provides PR link for review results

## Complete Safe Push Process

### Phase 1: Pre-commit Safety Checks

#### 1.1 Verify correct branch
```bash
# Check current branch
current_branch=$(git branch --show-current)
echo "Current branch: $current_branch"

# Should match pattern: feature/issue-XX-description
if [[ ! "$current_branch" =~ ^feature/issue-[0-9]+-.*$ ]]; then
  echo "⚠️ Branch name doesn't match pattern. Creating correct branch..."
  # Extract issue ID from working context
  issue_id="1" # Replace with actual issue ID
  git checkout -b "feature/issue-${issue_id}-smart-crop-implementation"
fi
```

#### 1.2 Check for personal images and model files
```bash
# List all staged files
echo "📋 Checking staged files..."
git status --porcelain

# Check for dangerous file types
dangerous_files=$(git status --porcelain | grep -E "\.(jpg|jpeg|png|gif|bmp|tiff|raw|heic|pkl|pt|pth|onnx|h5|pb|weights)$|test.*images?|sample.*photos?" || true)
if [ ! -z "$dangerous_files" ]; then
  echo "🚨 DANGER: Found sensitive files:"
  echo "$dangerous_files"
  echo "Unstaging dangerous files..."
  git reset HEAD *.jpg *.jpeg *.png *.gif *.pkl *.pt *.onnx
fi

# Search for privacy-sensitive patterns in staged files
echo "🔍 Scanning for privacy patterns..."
privacy_patterns=$(git diff --cached | grep -E "personal|private|test.*image|sample.*photo|face.*recognition|identity" || true)
if [ ! -z "$privacy_patterns" ]; then
  echo "⚠️ WARNING: Potential privacy-sensitive content found"
  echo "Review these matches carefully:"
  echo "$privacy_patterns" | head -10
fi
```

#### 1.3 Model file validation
```bash
# Check for accidentally staged model files
echo "🤖 Checking for model files..."
model_files=$(git status --porcelain | grep -E "\.(pkl|pt|pth|onnx|h5|pb|weights)$" || true)
if [ ! -z "$model_files" ]; then
  echo "⚠️ Model files detected - should use Git LFS:"
  echo "$model_files"
  echo "Consider: git lfs track '*.pkl' '*.pt' '*.onnx'"
fi
```

### Phase 2: Safe Staging

#### 2.1 Stage only safe files
```bash
# Reset everything first
git reset HEAD

# Add only code directories (no images or models)
echo "✅ Staging safe directories..."
git add photosight/ tests/ docs/ *.py *.md requirements.txt setup.py

# Show what will be committed
echo "📦 Files to be committed:"
git status --short
```

#### 2.2 Final safety review
```bash
# Check for any image data embedded in code
echo "🔍 Final privacy check..."
embedded_data=$(git diff --cached -- "*.py" "*.json" | grep -i "image.*data\|base64\|blob" || true)
if [ ! -z "$embedded_data" ]; then
  echo "⚠️ Found potential embedded image data - review carefully"
fi
```

### Phase 3: Commit and Push

#### 3.1 Create safe commit
```bash
# Analyze changes for commit message
if git diff --cached --name-only | grep -q "photosight/processing/"; then
  commit_type="feat"
  commit_scope="processing"
elif git diff --cached --name-only | grep -q "photosight/ai/"; then
  commit_type="feat"
  commit_scope="ai"
elif git diff --cached --name-only | grep -q "tests/"; then
  commit_type="test"
  commit_scope="tests"
elif git diff --cached --name-only | grep -q "docs/"; then
  commit_type="docs"
  commit_scope="docs"
else
  commit_type="fix"
  commit_scope="general"
fi

# Generate commit message
commit_msg="${commit_type}(${commit_scope}): Add description here"
echo "📝 Commit message: $commit_msg"

# Commit with message
git commit -m "$commit_msg"
```

#### 3.2 Push to remote
```bash
# Push the feature branch
git push -u origin $(git branch --show-current)
```

### Phase 4: Create PR with Code Review

#### 4.1 Create pull request
```bash
# Get the current branch name
branch_name=$(git branch --show-current)

# Create PR using GitHub CLI
pr_url=$(gh pr create \
  --title "[$branch_name] Photosight feature implementation" \
  --body "## Summary
- Added/Modified computer vision feature
- No personal images or test photos included
- Privacy-preserving implementation
- Follows photosight coding standards

## Test Plan
- [ ] Verified no personal images in changes
- [ ] Tested with synthetic/public domain images
- [ ] Validated crop suggestions are privacy-safe
- [ ] Checked model file handling

## Privacy Compliance
- [ ] No facial recognition of real people
- [ ] No personal photo data embedded
- [ ] Crop suggestions preserve privacy

🤖 Automated Claude Code Review will run on this PR" \
  --base main \
  --head "$branch_name")

echo "✅ PR created: $pr_url"
```

## Safety Checklist

### ✅ ALWAYS DO:
- Use feature branch pattern: `feature/issue-XX-description`
- Check for personal images before committing
- Stage only code files (photosight/, tests/, docs/)
- Review staged changes before commit
- Create descriptive commit messages
- Push to feature branch (never main)
- Create PR for code review

### 🚫 NEVER DO:
- Use `git add .` without checking for images
- Commit .jpg, .png, or other image files
- Include personal photos or test images
- Commit large model files without Git LFS
- Push directly to main branch
- Skip privacy checks
- Include facial recognition of real people

## Quick Commands

```bash
# Complete safe push workflow
./scripts/safe-push.sh

# Or manually:
git checkout -b feature/issue-XX-description
git add photosight/ tests/ docs/
git diff --cached | grep -E "image|photo|face"  # Privacy check
git commit -m "feat: Add privacy-safe feature"
git push -u origin feature/issue-XX-description
gh pr create --title "Safe CV feature" --body "Privacy-compliant implementation"
```

## Emergency Recovery

If you accidentally committed sensitive data:

```bash
# DO NOT PUSH!
# Reset the commit
git reset --soft HEAD~1

# Remove sensitive files
git reset HEAD *.jpg *.png *.pkl

# Recommit safely
git add photosight/ tests/
git commit -m "feat: Safe version without sensitive data"
```

## Code Review Integration

The PR will automatically trigger:
1. Claude Code Review via GitHub Actions
2. Privacy scanning for image data patterns
3. Computer vision compliance checks
4. Test suite execution for image processing

Monitor the PR for review results and address any findings before merging.