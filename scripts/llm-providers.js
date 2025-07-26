#!/usr/bin/env node

/**
 * LLM Provider Abstraction Layer for PhotoSight
 * Supports multiple LLM providers: Anthropic Claude, Google Gemini
 */

const { Anthropic } = require('@anthropic-ai/sdk');
const { GoogleGenAI } = require('@google/genai');
require('dotenv').config();

/**
 * Abstract base class for LLM providers
 */
class LLMProvider {
  constructor(name, apiKey) {
    this.name = name;
    this.apiKey = apiKey;
  }

  async generateContent(prompt, options = {}) {
    throw new Error('generateContent must be implemented by subclass');
  }

  isConfigured() {
    return !!this.apiKey || !!this.client;
  }
}

/**
 * Anthropic Claude provider implementation
 */
class AnthropicProvider extends LLMProvider {
  constructor(apiKey) {
    super('anthropic', apiKey);
    if (apiKey) {
      this.client = new Anthropic({ apiKey });
    }
  }

  async generateContent(prompt, options = {}) {
    if (!this.client) {
      throw new Error('Anthropic API key not configured');
    }

    const {
      model = 'claude-3-5-sonnet-20241022',
      maxTokens = 4000,
      temperature = 0.7
    } = options;

    try {
      const response = await this.client.messages.create({
        model,
        max_tokens: maxTokens,
        temperature,
        messages: [{
          role: 'user',
          content: prompt
        }]
      });

      return {
        content: response.content[0].text,
        model: model,
        provider: 'anthropic',
        usage: {
          inputTokens: response.usage?.input_tokens || 0,
          outputTokens: response.usage?.output_tokens || 0
        }
      };
    } catch (error) {
      throw new Error(`Anthropic API error: ${error.message}`);
    }
  }
}

/**
 * Google Gemini provider implementation
 */
class GeminiProvider extends LLMProvider {
  constructor(apiKey) {
    super('gemini', apiKey);
    // Try API key first, then fall back to ADC
    if (apiKey) {
      this.client = new GoogleGenAI({ apiKey });
    } else {
      // Use Application Default Credentials (gcloud auth)
      try {
        this.client = new GoogleGenAI({});
        this.usingADC = true;
      } catch (error) {
        console.warn('Failed to initialize Gemini with ADC:', error.message);
      }
    }
  }

  async generateContent(prompt, options = {}) {
    if (!this.client) {
      throw new Error('Gemini API key not configured');
    }

    const {
      model = 'gemini-2.5-flash',
      maxTokens = 4000,
      temperature = 0.7
    } = options;

    try {
      const response = await this.client.models.generateContent({
        model,
        contents: prompt,
        generationConfig: {
          maxOutputTokens: maxTokens,
          temperature
        }
      });

      return {
        content: response.text,
        model: model,
        provider: 'gemini',
        usage: {
          inputTokens: response.usage?.promptTokenCount || 0,
          outputTokens: response.usage?.candidatesTokenCount || 0
        }
      };
    } catch (error) {
      throw new Error(`Gemini API error: ${error.message}`);
    }
  }
}

/**
 * LLM Provider Factory and Manager
 */
class LLMProviderManager {
  constructor() {
    this.providers = new Map();
    this.defaultProvider = null;
    this._initializeProviders();
  }

  _initializeProviders() {
    // Initialize Anthropic provider
    const anthropicKey = process.env.ANTHROPIC_API_KEY;
    if (anthropicKey) {
      this.providers.set('anthropic', new AnthropicProvider(anthropicKey));
      if (!this.defaultProvider) {
        this.defaultProvider = 'anthropic';
      }
    }

    // Initialize Gemini provider (try API key first, then ADC)
    const geminiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_AI_API_KEY;
    const geminiProvider = new GeminiProvider(geminiKey);
    if (geminiProvider.isConfigured()) {
      this.providers.set('gemini', geminiProvider);
      if (!this.defaultProvider) {
        this.defaultProvider = 'gemini';
      }
    }
  }

  getProvider(name) {
    if (!name) {
      name = this.defaultProvider;
    }
    
    const provider = this.providers.get(name);
    if (!provider) {
      throw new Error(`Provider '${name}' not found or not configured`);
    }
    
    return provider;
  }

  getAvailableProviders() {
    return Array.from(this.providers.keys()).filter(name => 
      this.providers.get(name).isConfigured()
    );
  }

  async generateContent(prompt, options = {}) {
    const { provider: providerName, ...generateOptions } = options;
    const provider = this.getProvider(providerName);
    
    console.log(`🤖 Using ${provider.name} provider for content generation`);
    
    return await provider.generateContent(prompt, generateOptions);
  }

  setDefaultProvider(name) {
    if (!this.providers.has(name)) {
      throw new Error(`Provider '${name}' not found`);
    }
    this.defaultProvider = name;
  }

  getStatus() {
    const status = {};
    for (const [name, provider] of this.providers) {
      status[name] = {
        configured: provider.isConfigured(),
        isDefault: name === this.defaultProvider
      };
    }
    return status;
  }
}

// Export singleton instance
const llmManager = new LLMProviderManager();

module.exports = {
  LLMProviderManager,
  AnthropicProvider,
  GeminiProvider,
  llmManager
};