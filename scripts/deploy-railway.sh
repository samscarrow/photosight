#!/bin/bash

# Deploy Photosight to Railway.app
# Alternative to Vercel for API hosting

set -e

echo "🚂 Deploying Photosight to Railway.app..."

# Check if railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    curl -fsSL https://railway.app/install.sh | sh
fi

# Check if logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Please log in to Railway:"
    railway login
fi

# Load secrets from 1Password if available
if command -v op &> /dev/null && op vault list &> /dev/null; then
    echo "🔐 Loading secrets from 1Password..."
    
    if op item get "Anthropic API Key" &> /dev/null; then
        export ANTHROPIC_API_KEY=$(op read "op://Personal/Anthropic API Key/credential")
        echo "✅ Loaded ANTHROPIC_API_KEY"
    else
        echo "⚠️ Anthropic API Key not found in 1Password"
        echo "Please enter your Anthropic API key:"
        read -s ANTHROPIC_API_KEY
        export ANTHROPIC_API_KEY
    fi
else
    echo "⚠️ 1Password CLI not available or not logged in"
    echo "Please enter your Anthropic API key:"
    read -s ANTHROPIC_API_KEY
    export ANTHROPIC_API_KEY
fi

# Check if project exists
if ! railway status &> /dev/null; then
    echo "🆕 Creating new Railway project..."
    railway init photosight
fi

# Set environment variables
echo "🔧 Setting environment variables..."
railway variables set ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
railway variables set NODE_ENV="production"
railway variables set MAX_IMAGE_SIZE_MB="10"
railway variables set ENABLE_TELEMETRY="false"
railway variables set STORE_TEMP_FILES="false"

# Deploy
echo "🚀 Deploying to Railway..."
railway up

# Get deployment URL
deployment_url=$(railway domain)
if [ ! -z "$deployment_url" ]; then
    echo "✅ Deployment successful!"
    echo "🌐 URL: https://$deployment_url"
    echo "📋 API endpoints:"
    echo "   - Health: https://$deployment_url/health"
    echo "   - Crop suggestion: https://$deployment_url/api/suggest-crop"
    echo "   - Docs: https://$deployment_url/docs"
else
    echo "⚠️ Deployment completed but URL not available yet"
    echo "Check Railway dashboard: https://railway.app/dashboard"
fi

echo "🎉 Railway deployment complete!"