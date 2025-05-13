/**
 * Configuration for Social Media LLM features
 * 
 * This file contains settings for:
 * - Custom LLM integration
 * - OpenAI API fallback
 * - Feature toggles
 * - UI integration points
 */

// Base model settings
export interface ModelSettings {
  /** Local model path or OpenAI model name */
  modelId: string;
  /** Maximum number of tokens to generate */
  maxTokens: number;
  /** Temperature for generation (higher = more creative) */
  temperature: number;
  /** Request timeout in milliseconds */
  timeout: number;
  /** Whether model supports streaming responses */
  supportsStreaming: boolean;
}

// Feature settings
export interface FeatureSettings {
  /** Enable multilingual rewriting */
  enableMultilingual: boolean;
  /** Enable engagement prediction */
  enableEngagementPrediction: boolean;
  /** Enable sentiment analysis */
  enableSentimentAnalysis: boolean;
  /** Enable hashtag suggestions */
  enableHashtagSuggestions: boolean;
  /** Enable viral potential prediction */
  enableViralPotential: boolean;
  /** Enable topic classification */
  enableTopicClassification: boolean;
}

// API settings
export interface ApiSettings {
  /** OpenAI API key (for fallback) */
  openAiApiKey: string;
  /** Base URL for custom model API */
  customModelBaseUrl: string;
  /** Whether to use local custom model */
  useLocalModel: boolean;
  /** Path to local model files */
  localModelPath: string;
  /** Whether to save API calls for reuse */
  cacheResponses: boolean;
  /** Maximum retries for API calls */
  maxRetries: number;
}

// UI integration settings
export interface UiSettings {
  /** UI Locations where LLM features are available */
  integrationPoints: {
    postEditor: boolean;
    analyticsTab: boolean;
    profilePage: boolean;
    schedulerPage: boolean;
  };
  /** Default language for multilingual features */
  defaultLanguage: string;
  /** Display settings */
  display: {
    showModelInfo: boolean;
    showApiKeyInput: boolean;
    showAdvancedOptions: boolean;
    showMetricsExplanations: boolean;
  };
}

// Complete LLM configuration
export interface LlmConfig {
  model: ModelSettings;
  features: FeatureSettings;
  api: ApiSettings;
  ui: UiSettings;
  /** Whether to use the custom trained model */
  useCustomModel: boolean;
  /** Version of this configuration */
  version: string;
}

// Default configuration
const defaultConfig: LlmConfig = {
  model: {
    modelId: "gpt-4o-mini", // Default to OpenAI model as fallback
    maxTokens: 2048,
    temperature: 0.7,
    timeout: 30000,
    supportsStreaming: true,
  },
  features: {
    enableMultilingual: true,
    enableEngagementPrediction: true,
    enableSentimentAnalysis: true,
    enableHashtagSuggestions: true,
    enableViralPotential: true,
    enableTopicClassification: true, // Disabled by default as it's experimental
  },
  api: {
    openAiApiKey: "", // TODO: "will change to be my own api key to charge the user if they want their own api key ",
    customModelBaseUrl: "http://localhost:8000/api",
    useLocalModel: false, // Default to API mode
    localModelPath: "./models/social-llm",
    cacheResponses: true,
    maxRetries: 3,
  },
  ui: {
    integrationPoints: {
      postEditor: true,
      analyticsTab: true,
      profilePage: true,
      schedulerPage: true,
    },
    defaultLanguage: "en",
    display: {
      showModelInfo: true,
      showApiKeyInput: true,
      showAdvancedOptions: true,
      showMetricsExplanations: true,
    },
  },
  useCustomModel: false, // Default to false until custom model is available TODO: implement the custom model for the user to use and may charge the user to use the custom model 
  version: "1.0.0",
};

// Custom model configuration
export const customModelConfig: ModelSettings = {
  modelId: "social-llm-v1",
  maxTokens: 2028,
  temperature: 0.8,
  timeout: 60000,
  supportsStreaming: false,
};

// OpenAI model configurations
export const openAiModels: Record<string, ModelSettings> = {
  "gpt-4o": {
    modelId: "gpt-4o",
    maxTokens: 4000,
    temperature: 0.7,
    timeout: 60000,
    supportsStreaming: true,
  },
  "gpt-4o-mini": {
    modelId: "gpt-4o-mini",
    maxTokens: 4000,
    temperature: 0.7,
    timeout: 30000,
    supportsStreaming: true,
  },
  "gpt-3.5-turbo": {
    modelId: "gpt-3.5-turbo",
    maxTokens: 4000,
    temperature: 0.7,
    timeout: 30000,
    supportsStreaming: true,
  },
};

// Available languages for multilingual features
export const availableLanguages = [
  { code: "en", name: "English" },
  { code: "es", name: "Spanish" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "it", name: "Italian" },
  { code: "pt", name: "Portuguese" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
  { code: "zh", name: "Chinese" },
  { code: "ar", name: "Arabic" },
  { code: "hi", name: "Hindi" },
  { code: "ru", name: "Russian" },
];

// Function to get the current configuration
/**
 * Retrieves and merges LLM configuration from localStorage with default settings.
 */
export function getLlmConfig(): LlmConfig {
  // Try to load config from localStorage
  try {
    const storedConfig = localStorage.getItem("llm_config");
    if (storedConfig) {
      const parsedConfig = JSON.parse(storedConfig) as LlmConfig;
      
      // Add any missing properties from default config
      const mergedConfig = {
        ...defaultConfig,
        ...parsedConfig,
        model: { ...defaultConfig.model, ...parsedConfig.model },
        features: { ...defaultConfig.features, ...parsedConfig.features },
        api: { ...defaultConfig.api, ...parsedConfig.api },
        ui: { ...defaultConfig.ui, ...parsedConfig.ui },
        version: defaultConfig.version, // Always use latest version
      };
      
      return mergedConfig;
    }
  } catch (error) {
    console.error("Error loading LLM config:", error);
  }
  
  return defaultConfig;
}

// Function to save configuration
/**
 * Saves the given LlmConfig to localStorage.
 */
export function saveLlmConfig(config: LlmConfig): void {
  try {
    localStorage.setItem("llm_config", JSON.stringify(config));
  } catch (error) {
    console.error("Error saving LLM config:", error);
  }
}

// Function to update a specific part of the configuration
/**
 * Updates the LLM configuration with provided changes and saves it.
 */
export function updateLlmConfig(
  updates: Partial<LlmConfig> | ((config: LlmConfig) => LlmConfig)
): LlmConfig {
  const currentConfig = getLlmConfig();
  
  let newConfig: LlmConfig;
  if (typeof updates === "function") {
    newConfig = updates(currentConfig);
  } else {
    newConfig = {
      ...currentConfig,
      ...updates,
      model: { ...currentConfig.model, ...(updates.model || {}) },
      features: { ...currentConfig.features, ...(updates.features || {}) },
      api: { ...currentConfig.api, ...(updates.api || {}) },
      ui: { ...currentConfig.ui, ...(updates.ui || {}) },
    };
  }
  
  saveLlmConfig(newConfig);
  return newConfig;
}

// Function to reset configuration to defaults
/**
 * Resets the LLM configuration by removing it from local storage and returning the default configuration.
 */
export function resetLlmConfig(): LlmConfig {
  localStorage.removeItem("llm_config");
  return { ...defaultConfig };
}

// Get the currently configured model settings
/**
 * Retrieves the current model settings based on configuration.
 */
export function getCurrentModelSettings(): ModelSettings {
  const config = getLlmConfig();
  
  if (config.useCustomModel) {
    return customModelConfig;
  }
  
  // Try to get the selected OpenAI model
  const openAiModel = openAiModels[config.model.modelId];
  if (openAiModel) {
    return openAiModel;
  }
  
  // Fallback to the config model settings
  return config.model;
}

// Export the default configuration
export default defaultConfig;

