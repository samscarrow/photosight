# Photosight Git Architecture Summary

## Overview
Successfully implemented git architecture from bvaadmin project with Claude Code Review integration for the photosight computer vision system.

## Components Implemented

### 1. Claude Commands Structure (`.claude/commands/`)
- **push.md**: Simple git push with Claude Code Review
- **safe-push.md**: Comprehensive safe push workflow with privacy checks

### 2. Git Security (`git-secrets`)
- Installed and configured git-secrets hooks
- Added patterns for:
  - AWS credentials
  - Anthropic API keys
  - OpenAI API keys
  - General API key patterns

### 3. 1Password CLI Integration
- **setup-secrets.sh**: Interactive script to store secrets in 1Password
- **load-secrets.sh**: Script to load secrets from 1Password
- Configured for Anthropic API, OpenAI API, and Vercel tokens

### 4. Claude Code Review System
- **claude-review-pr.js**: Node.js script for AI-powered code review
- **GitHub Actions workflow**: Automated PR review trigger
- Privacy-specific checks for computer vision code:
  - Image file detection
  - Personal data pattern scanning
  - Model file validation
  - Embedded data detection

### 5. Safe Push Workflow
- **safe-push.sh**: Interactive bash script implementing complete workflow
- Privacy checks specific to photosight:
  - Blocks personal images and test photos
  - Validates model file handling
  - Scans for facial recognition patterns
  - Enforces branch naming conventions

### 6. Alternative API Hosting
- **Railway.app configuration**: Alternative to maxed-out Vercel
- **Dockerfile**: Privacy-first container setup
- **deploy-railway.sh**: Automated deployment script

## Security Features

### Git Hooks
- Pre-commit: Prevents secrets from being committed
- Commit-msg: Validates commit message format
- Prepare-commit-msg: Prepares standardized commit messages

### Privacy Protection
- Automatic image file detection and blocking
- Personal data pattern scanning
- Facial recognition content prevention
- Embedded data detection

### Secret Management
- 1Password CLI integration
- Environment variable management
- Never commit secrets to repository

## Usage Workflows

### Daily Development
```bash
# Load secrets
source scripts/load-secrets.sh

# Make changes, then safe push
/safe-push
```

### Initial Setup
```bash
# Setup secrets management
./scripts/setup-secrets.sh

# Configure git secrets
git secrets --install
git secrets --register-aws
```

### Deployment
```bash
# Deploy to Railway (Vercel alternative)
./scripts/deploy-railway.sh
```

## Claude Code Review Integration

### Trigger Policy
- **Automatic**: Runs on all PR creation/updates
- **Manual**: Can be triggered via workflow dispatch
- **Blocking**: PRs with hardcoded secrets are blocked

### Review Focus Areas
1. **Privacy Compliance**: No personal data retention
2. **Security**: Secret management and API security
3. **Performance**: Algorithm efficiency and memory usage
4. **Code Quality**: Maintainability and testing

### Review Categories
- 🚨 **CRITICAL**: Security/privacy violations
- ⚠️ **IMPORTANT**: Performance/architecture issues  
- 💡 **SUGGESTION**: Code style and optimizations

## Branch Strategy
- **Pattern**: `feature/issue-XX-description`
- **Protection**: Cannot push directly to main
- **Review Required**: All PRs get Claude review

## Environment Variables
- `ANTHROPIC_API_KEY`: Claude API access
- `OPENAI_API_KEY`: Optional for comparison models
- `VERCEL_TOKEN`: Deployment token
- `ENABLE_TELEMETRY`: Privacy setting (default: false)
- `STORE_TEMP_FILES`: Privacy setting (default: false)

## Files Created
```
.claude/
├── commands/
│   ├── push.md
│   └── safe-push.md
.github/
└── workflows/
    └── claude-code-review.yml
scripts/
├── claude-review-pr.js
├── setup-secrets.sh
├── load-secrets.sh
├── safe-push.sh
└── deploy-railway.sh
railway.json
Dockerfile
.env.example
```

## Next Steps
1. Run `./scripts/setup-secrets.sh` to configure secrets
2. Create first feature branch: `git checkout -b feature/issue-1-smart-crop`
3. Use `/safe-push` command for all commits
4. Monitor Claude Code Review results on PRs