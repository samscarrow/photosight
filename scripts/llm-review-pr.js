#!/usr/bin/env node

/**
 * LLM Code Review for Photosight Pull Requests
 * Analyzes PR changes and provides intelligent code review using configurable LLM providers
 * Supports: Anthropic Claude, Google Gemini
 */

const { llmManager } = require('./llm-providers');
const { execSync } = require('child_process');
const fs = require('fs');
require('dotenv').config();

async function reviewPullRequest(prNumber, options = {}) {
  const { provider = null } = options;
  
  // Check available providers
  const availableProviders = llmManager.getAvailableProviders();
  if (availableProviders.length === 0) {
    throw new Error('No LLM providers configured. Please set ANTHROPIC_API_KEY or GEMINI_API_KEY');
  }
  
  const selectedProvider = provider || availableProviders[0];
  console.log(`🤖 Starting LLM Code review for Photosight PR #${prNumber} using ${selectedProvider}`);

  try {
    // Get PR information
    const prInfo = JSON.parse(execSync(`gh pr view ${prNumber} --json title,body,author,files,commits`, { encoding: 'utf8' }));
    
    // Get diff content
    const diffOutput = execSync(`git diff origin/main...HEAD`, { encoding: 'utf8' });
    
    // Get changed files
    const changedFiles = execSync(`git diff --name-only origin/main...HEAD`, { encoding: 'utf8' })
      .split('\n').filter(f => f.trim());
    
    // Read relevant file contents for context
    const fileContents = {};
    for (const file of changedFiles.slice(0, 10)) { // Limit to first 10 files
      try {
        if (fs.existsSync(file)) {
          fileContents[file] = fs.readFileSync(file, 'utf8');
        }
      } catch (error) {
        console.warn(`Could not read file ${file}: ${error.message}`);
      }
    }

    // Analyze PR type and scope for Photosight
    const isAIModel = changedFiles.some(f => 
      f.includes('photosight/ai/') || 
      f.includes('models/') || 
      f.includes('.pkl') || 
      f.includes('.pt') || 
      f.includes('.onnx')
    );
    
    const isImageProcessing = changedFiles.some(f => 
      f.includes('photosight/processing/') ||
      f.includes('crop') ||
      f.includes('geometry')
    );
    
    const isComputerVision = changedFiles.some(f => 
      f.includes('vision/') ||
      f.includes('detection/') ||
      f.includes('analysis/')
    );
    
    const hasImageFiles = changedFiles.some(f => 
      f.match(/\.(jpg|jpeg|png|gif|bmp|tiff|raw|heic)$/i)
    );
    
    const prType = isAIModel ? 'ai-model' : 
                   isImageProcessing ? 'image-processing' : 
                   isComputerVision ? 'computer-vision' : 'application';

    // Prepare enhanced PR review prompt
    const reviewPrompt = `
Review this ${prType} Pull Request for the Photosight computer vision system:

**PR #${prNumber}:** ${prInfo.title}
**Author:** ${prInfo.author.login}
**Type:** ${prType.toUpperCase()} (${changedFiles.length} files changed)
**Files:** ${changedFiles.slice(0, 5).join(', ')}${changedFiles.length > 5 ? ` +${changedFiles.length - 5} more` : ''}
${hasImageFiles ? '⚠️ **CONTAINS IMAGE FILES**' : ''}

**Description:**
${prInfo.body || 'No description provided'}

**Context:** Photosight is a privacy-preserving computer vision system for intelligent photo cropping and composition analysis.

**Diff (first 8000 chars):**
\`\`\`diff
${diffOutput.slice(0, 8000)}${diffOutput.length > 8000 ? '\n... (truncated for length)' : ''}
\`\`\`

**Review Focus for ${prType} PR:**

${prType === 'ai-model' ? `
🤖 **AI Model Review:**
- Model file size and Git LFS usage
- Privacy-preserving inference pipelines
- Model bias and fairness considerations
- Performance optimization and memory usage
` : prType === 'image-processing' ? `
🖼️ **Image Processing Review:**
- Privacy-safe image handling (no personal data retention)
- Efficient algorithms and memory management
- Cross-platform compatibility
- Edge case handling for various image formats
` : prType === 'computer-vision' ? `
👁️ **Computer Vision Review:**
- Detection accuracy and false positive rates
- Privacy compliance (no facial recognition of real people)
- Algorithm efficiency and real-time performance
- Integration with cropping and composition systems
` : `
💻 **Application Review:**
- Code quality and maintainability
- Privacy and security compliance
- User experience and accessibility
- Testing coverage and reliability
`}

**Priority Levels:**
🚨 **CRITICAL** - Privacy violations, model security issues, data retention risks
⚠️ **IMPORTANT** - Performance issues, algorithm accuracy, memory leaks
💡 **SUGGESTION** - Code style, optimizations, best practices

**Photosight Specific:**
- Privacy-first design (no personal photo storage)
- Crop suggestions preserve subject privacy
- Model inference runs locally when possible
- No unauthorized facial recognition
- Synthetic/public domain test data only

**Image File Policy:**
${hasImageFiles ? '🚨 **CRITICAL**: Image files detected in PR. Verify:' : '✅ No image files detected'}
${hasImageFiles ? `
- Are these public domain or synthetic test images?
- Is there explicit permission for any real photos?
- Are faces/people anonymized or removed?
- Could these be replaced with generated test data?
` : ''}

**Required Response Format:**
1. **Overall Recommendation:** APPROVE / REQUEST_CHANGES / COMMENT
2. **Priority-based feedback** using 🚨⚠️💡 indicators
3. **Specific actionable improvements** - focus on what to change and why
4. **Privacy compliance check** - data handling and retention policies
5. **Image file assessment** (if applicable)

Skip generic observations - provide actionable feedback only.
`;

    // Get LLM review
    const response = await llmManager.generateContent(reviewPrompt, {
      provider: selectedProvider,
      maxTokens: 4000,
      temperature: 0.7
    });

    const reviewContent = response.content;
    console.log(`✨ Review generated using ${response.provider} (${response.model})`);

    // Post review comment via GitHub CLI
    const reviewFile = `/tmp/claude-review-${prNumber}.md`;
    fs.writeFileSync(reviewFile, reviewContent);

    // Post the review
    execSync(`gh pr review ${prNumber} --comment --body-file ${reviewFile}`, { stdio: 'inherit' });

    // Clean up
    fs.unlinkSync(reviewFile);

    console.log(`✅ LLM Code review posted for PR #${prNumber} (${response.provider})`);

    // Extract overall assessment
    const assessment = reviewContent.toLowerCase().includes('request_changes') ? 'REQUEST_CHANGES' :
                     reviewContent.toLowerCase().includes('approve') ? 'APPROVE' : 'COMMENT';
    
    return {
      prNumber,
      assessment,
      reviewPosted: true,
      hasImageFiles,
      provider: response.provider,
      model: response.model,
      usage: response.usage
    };

  } catch (error) {
    console.error(`❌ Error reviewing PR #${prNumber}:`, error.message);
    
    // Post error comment
    const errorComment = `🤖 LLM Code Review Error

Sorry, I encountered an error while reviewing this PR:
\`\`\`
${error.message}
\`\`\`

Please check the workflow logs for more details.`;

    try {
      execSync(`gh pr comment ${prNumber} --body "${errorComment}"`, { stdio: 'inherit' });
    } catch (commentError) {
      console.error('Failed to post error comment:', commentError.message);
    }

    throw error;
  }
}

// Main execution
async function main() {
  const args = process.argv.slice(2);
  const prNumber = args[0];
  const provider = args[1]; // Optional provider argument
  
  if (!prNumber) {
    console.error('❌ Usage: node claude-review-pr.js <pr_number> [provider]');
    console.error('   Providers: anthropic, gemini');
    console.error('   Environment variables: ANTHROPIC_API_KEY, GEMINI_API_KEY');
    process.exit(1);
  }

  // Check available providers
  const availableProviders = llmManager.getAvailableProviders();
  if (availableProviders.length === 0) {
    console.error('❌ No LLM providers configured');
    console.error('💡 Set ANTHROPIC_API_KEY and/or GEMINI_API_KEY environment variables');
    console.error('💡 Anthropic: op read "op://Personal/Anthropic API/credential"');
    console.error('💡 Gemini: op read "op://Personal/Google AI API/credential"');
    process.exit(1);
  }

  console.log(`🔧 Available providers: ${availableProviders.join(', ')}`);
  
  if (provider && !availableProviders.includes(provider)) {
    console.error(`❌ Provider '${provider}' not available or not configured`);
    console.error(`💡 Available: ${availableProviders.join(', ')}`);
    process.exit(1);
  }

  try {
    const result = await reviewPullRequest(prNumber, { provider });
    console.log('🎉 Review completed:', result);
    
    if (result.hasImageFiles) {
      console.log('⚠️ WARNING: This PR contains image files - manual privacy review recommended');
    }
    
    process.exit(0);
  } catch (error) {
    console.error('❌ Review failed:', error.message);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}

module.exports = { reviewPullRequest };