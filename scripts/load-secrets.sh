#!/bin/bash

# Load secrets from 1Password for Photosight development
# Usage: source scripts/load-secrets.sh

echo "🔐 Loading secrets from 1Password..."

# Check if op CLI is available and logged in
if ! command -v op &> /dev/null; then
    echo "❌ 1Password CLI not found. Install with: brew install 1password-cli"
    return 1 2>/dev/null || exit 1
fi

if ! op vault list &> /dev/null; then
    echo "❌ Please log in to 1Password: op signin"
    return 1 2>/dev/null || exit 1
fi

# Load Anthropic API Key from existing item
if op item get "anthropic-api-key" &> /dev/null; then
    export ANTHROPIC_API_KEY=$(op item get "anthropic-api-key" --fields credential --reveal)
    echo "✅ Loaded ANTHROPIC_API_KEY"
else
    echo "⚠️ Anthropic API Key not found in 1Password"
fi

# Privacy-first defaults
export ENABLE_TELEMETRY=false
export STORE_TEMP_FILES=false
export MAX_IMAGE_SIZE_MB=10

echo "🎉 Secrets loaded successfully!"
echo "💡 To use: source scripts/load-secrets.sh"