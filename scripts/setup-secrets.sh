#!/bin/bash

# Setup script for Photosight secrets management
# Uses 1Password CLI (op) for secure secret storage

set -e

echo "🔐 Setting up Photosight secrets management..."

# Check if op CLI is available
if ! command -v op &> /dev/null; then
    echo "❌ 1Password CLI (op) not found. Install with: brew install 1password-cli"
    exit 1
fi

# Check if logged in to 1Password
if ! op vault list &> /dev/null; then
    echo "❌ Please log in to 1Password CLI first: op signin"
    exit 1
fi

# Function to create or update secret in 1Password
create_or_update_secret() {
    local title="$1"
    local field_name="$2"
    local vault="Personal"
    
    echo "📝 Setting up secret: $title"
    
    # Check if item exists
    if op item get "$title" --vault="$vault" &> /dev/null; then
        echo "✅ Secret '$title' already exists in 1Password"
    else
        echo "🆕 Creating new secret '$title' in 1Password..."
        echo "Please enter the value for $field_name:"
        read -s secret_value
        
        op item create \
            --category="API Credential" \
            --title="$title" \
            --vault="$vault" \
            --field="label=$field_name,type=concealed,value=$secret_value"
        
        echo "✅ Secret '$title' created successfully"
    fi
}

# Setup Anthropic API Key
create_or_update_secret "Anthropic API Key" "credential"

# Setup OpenAI API Key (if needed for comparison models)
echo "Do you need to store an OpenAI API key? (y/n)"
read -r setup_openai
if [[ $setup_openai == "y" || $setup_openai == "Y" ]]; then
    create_or_update_secret "OpenAI API Key" "credential"
fi

# Setup Vercel credentials (if using for API hosting)
echo "Do you need to store Vercel credentials? (y/n)"
read -r setup_vercel
if [[ $setup_vercel == "y" || $setup_vercel == "Y" ]]; then
    create_or_update_secret "Vercel Token" "token"
fi

# Create .env.example file
echo "📄 Creating .env.example file..."
cat > .env.example << 'EOF'
# Photosight Environment Variables
# Use 1Password CLI to retrieve actual values:
# export ANTHROPIC_API_KEY=$(op read "op://Personal/Anthropic API Key/credential")
# export OPENAI_API_KEY=$(op read "op://Personal/OpenAI API Key/credential")
# export VERCEL_TOKEN=$(op read "op://Personal/Vercel Token/token")

# API Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Hosting Configuration
VERCEL_TOKEN=your_vercel_token_here

# Privacy Settings
ENABLE_TELEMETRY=false
STORE_TEMP_FILES=false
MAX_IMAGE_SIZE_MB=10

# Development Settings
DEBUG=false
LOG_LEVEL=info
EOF

# Create secret loading script
echo "📄 Creating load-secrets.sh script..."
cat > scripts/load-secrets.sh << 'EOF'
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

# Load Anthropic API Key
if op item get "Anthropic API Key" &> /dev/null; then
    export ANTHROPIC_API_KEY=$(op read "op://Personal/Anthropic API Key/credential")
    echo "✅ Loaded ANTHROPIC_API_KEY"
else
    echo "⚠️ Anthropic API Key not found in 1Password"
fi

# Load OpenAI API Key (optional)
if op item get "OpenAI API Key" &> /dev/null; then
    export OPENAI_API_KEY=$(op read "op://Personal/OpenAI API Key/credential")
    echo "✅ Loaded OPENAI_API_KEY"
fi

# Load Vercel Token (optional)
if op item get "Vercel Token" &> /dev/null; then
    export VERCEL_TOKEN=$(op read "op://Personal/Vercel Token/token")
    echo "✅ Loaded VERCEL_TOKEN"
fi

# Privacy-first defaults
export ENABLE_TELEMETRY=false
export STORE_TEMP_FILES=false
export MAX_IMAGE_SIZE_MB=10

echo "🎉 Secrets loaded successfully!"
echo "💡 To use: source scripts/load-secrets.sh"
EOF

chmod +x scripts/load-secrets.sh

# Add .env to .gitignore
echo "📄 Updating .gitignore..."
if ! grep -q "^\.env$" .gitignore 2>/dev/null; then
    echo ".env" >> .gitignore
fi

if ! grep -q "^\.env\.local$" .gitignore 2>/dev/null; then
    echo ".env.local" >> .gitignore
fi

echo "✅ Secrets management setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Source the secrets: source scripts/load-secrets.sh"
echo "2. Test Claude review: node scripts/claude-review-pr.js --test"
echo "3. Create your first PR: /safe-push"
echo ""
echo "🔒 Your secrets are safely stored in 1Password and will never be committed to git."