#!/usr/bin/env node

/**
 * Test script for LLM providers in PhotoSight
 * Tests both Anthropic Claude and Google Gemini integration
 */

const { llmManager } = require('./llm-providers');
require('dotenv').config();

async function testProviders() {
  console.log('🔍 Testing LLM Providers for PhotoSight\n');
  
  // Check provider status
  const status = llmManager.getStatus();
  console.log('📊 Provider Status:');
  for (const [name, info] of Object.entries(status)) {
    console.log(`  ${name}: ${info.configured ? '✅ Configured' : '❌ Not configured'}${info.isDefault ? ' (default)' : ''}`);
  }
  
  const availableProviders = llmManager.getAvailableProviders();
  console.log(`\n🔧 Available providers: ${availableProviders.join(', ')}`);
  
  if (availableProviders.length === 0) {
    console.log('\n❌ No providers configured. Set ANTHROPIC_API_KEY and/or GEMINI_API_KEY');
    return;
  }
  
  console.log('\n🧪 Testing providers...\n');
  
  // Test prompt for PhotoSight
  const testPrompt = `Analyze this code change for PhotoSight:

\`\`\`python
def process_image(image_path):
    # Process image for privacy-preserving crop suggestions
    img = cv2.imread(image_path)
    return analyze_composition(img)
\`\`\`

Provide a brief technical review focusing on:
1. Privacy compliance
2. Code quality
3. One suggestion for improvement

Keep response under 100 words.`;

  // Test each provider
  for (const providerName of availableProviders) {
    try {
      console.log(`🤖 Testing ${providerName}...`);
      
      const startTime = Date.now();
      const response = await llmManager.generateContent(testPrompt, {
        provider: providerName,
        maxTokens: 200,
        temperature: 0.5
      });
      const duration = Date.now() - startTime;
      
      console.log(`✅ ${providerName} response (${duration}ms):`);
      console.log(`   Model: ${response.model}`);
      console.log(`   Tokens: ${response.usage.inputTokens} in, ${response.usage.outputTokens} out`);
      console.log(`   Preview: ${response.content.substring(0, 100)}...`);
      console.log('');
      
    } catch (error) {
      console.log(`❌ ${providerName} failed: ${error.message}\n`);
    }
  }
  
  console.log('🎉 Provider testing complete!');
}

// Test provider switching
async function testProviderSwitching() {
  const availableProviders = llmManager.getAvailableProviders();
  
  if (availableProviders.length < 2) {
    console.log('⚠️ Need at least 2 providers configured to test switching');
    return;
  }
  
  console.log('\n🔄 Testing provider switching...');
  
  for (const provider of availableProviders) {
    llmManager.setDefaultProvider(provider);
    console.log(`Set default to: ${provider}`);
    
    const response = await llmManager.generateContent('Hello! What provider am I using?', {
      maxTokens: 50
    });
    
    console.log(`Response from ${response.provider}: ${response.content.substring(0, 50)}...`);
  }
}

async function main() {
  try {
    await testProviders();
    await testProviderSwitching();
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main().catch(console.error);
}