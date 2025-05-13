/* eslint-disable @typescript-eslint/no-explicit-any */
import { toast } from "@/components/ui/sonner";

const API_URL = "https://api.openai.com/v1/chat/completions";

// Type definitions
export interface PostAnalysis {
  score: number;
  engagementScore: number;
  readabilityScore: number;
  viralityScore: number;
  algorithmFriendly: string[];
  optimizationOpportunities: string[];
  optimizationTips: string[];
  keyFactors: string[];
  bestPostingTime: {
    daily: string;
    peakDays: string;
  };
  targetAudience: string;
  keywords: string;
  sentiment: {
    tone: string;
    score: number;
    emotions: string[];
    isPositive: boolean;
    isNegative: boolean;
    isNeutral: boolean;
  };
  hashtags: {
    trending: string[];
    recommended: string[];
    reachPotential: string;
    relevanceScore: number;
  };
}

export interface EngagementMetrics {
  engagementRate: number;
  likeRate: number;
  repostRate: number;
  replyRate: number;
  performanceText: string;
}

export type OpenAIModel = "gpt-4o-mini" | "gpt-4o" | "gpt-4.5-preview";

// Define supported languages and their information
export interface LanguageInfo {
  code: string;
  name: string;
  characterLimit?: number;
  rtl?: boolean;
  localHashtagPrefix?: string;
}

// Common languages supported for rewriting
export const SUPPORTED_LANGUAGES: LanguageInfo[] = [
  { code: "en", name: "English", characterLimit: 280 },
  { code: "es", name: "Spanish", characterLimit: 280 },
  { code: "fr", name: "French", characterLimit: 280 },
  { code: "de", name: "German", characterLimit: 280 },
  { code: "it", name: "Italian", characterLimit: 280 },
  { code: "pt", name: "Portuguese", characterLimit: 280 },
  { code: "ja", name: "Japanese", characterLimit: 140 },
  { code: "ko", name: "Korean", characterLimit: 140 }, 
  { code: "zh", name: "Chinese", characterLimit: 140 }, 
  { code: "ar", name: "Arabic", characterLimit: 280, rtl: true },
  { code: "hi", name: "Hindi", characterLimit: 280 },
  { code: "ru", name: "Russian", characterLimit: 280 }
];

export interface MultilingualOptions {
  language?: string; // ISO language code (e.g., "en", "es")
  adaptForCulture?: boolean; // Adapt content for cultural context
  preserveHashtags?: boolean; // Keep original hashtags
  preserveMentions?: boolean; // Keep original @mentions
  regionalVariant?: string; // Regional variant (e.g., "US" for en-US)
}

// Helper function to safely parse OpenAI JSON responses
const safeJsonParse = (jsonString: string): any => {
  try {
    // Sometimes OpenAI adds backticks or other markdown to the JSON
    const cleanJsonString = jsonString
      .replace(/^```json/, '')
      .replace(/^```/, '')
      .replace(/```$/, '')
      .trim();
      
    return JSON.parse(cleanJsonString);
  } catch (error) {
    console.error("JSON parsing error:", error);
    console.log("Problematic JSON:", jsonString);
    throw new Error("Failed to parse analysis results");
  }
};

// Helper to get language-specific character limits and settings
/**
 * Retrieves language settings based on the provided language code.
 */
const getLanguageSettings = (languageCode?: string): LanguageInfo => {
  if (!languageCode) return { code: "en", name: "English", characterLimit: 280 };
  const language = SUPPORTED_LANGUAGES.find(lang => lang.code === languageCode);
  return language || { code: languageCode, name: languageCode, characterLimit: 280 };
};

const formatHashtagsForLanguage = (hashtags: string[], languageCode: string): string[] => {
  // Most languages use # for hashtags
  const prefix = "#";
  
  return hashtags.map(tag => {
    // Remove any existing hashtag symbols first
    const cleanTag = tag.replace(/^[#＃]/, '');
    return `${prefix}${cleanTag}`;
  });
};

/**
 * Analyzes a post using an AI model and returns a detailed analysis object.
 */
export const analyzePost = async (
post: string, includesMedia: boolean, apiKey: string, selectedModel: string, industry: string, goal: string, model: OpenAIModel = "gpt-4o-mini", language?: string): Promise<PostAnalysis> => {
  try {
    // Determine the language if not specified
    const detectedLanguage = language || detectLanguage(post);
    const languageInfo = getLanguageSettings(detectedLanguage);
    
    toast.info(`Analyzing your post${language ? ` (${languageInfo.name})` : ''}...`);
    
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model,
        messages: [
          {
            role: "system",
            content: `You are an expert social media analyst specializing in X (Twitter) posts. 
            ${language ? `Your analysis should be focused on content in ${languageInfo.name}.` : ''}
            Analyze the provided post and output a JSON object that includes: 
            1. An overall score from 0-100 as "score"
            2. An engagement score from 0-100 as "engagementScore"
            3. A readability score from 0-100 as "readabilityScore" 
            4. A virality score from 0-100 as "viralityScore"
            5. Algorithm friendly features as "algorithmFriendly" (array of strings)
            6. Optimization opportunities as "optimizationOpportunities" (array of strings)
            7. Optimization tips as "optimizationTips" (array of strings)
            8. Key algorithm factors as "keyFactors" (array of strings)
            9. Best posting time information as "bestPostingTime" (object with daily and peakDays properties)
            10. Target audience description as "targetAudience" (string)
            11. Optimal keywords for the post as "keywords" (comma-separated string)
            12. A sentiment analysis object as "sentiment" containing:
               - tone (string): overall emotional tone (e.g. positive, negative, neutral, sarcastic, enthusiastic)
               - score (number): 0-100 sentiment score where 0 is very negative, 50 is neutral, 100 is very positive
               - emotions (string[]): array of primary emotions detected
               - isPositive (boolean): true if sentiment is generally positive
               - isNegative (boolean): true if sentiment is generally negative
               - isNeutral (boolean): true if sentiment is generally neutral
            13. A hashtag analysis object as "hashtags" containing:
               - trending (string[]): array of hashtags in the post that are currently trending
               - recommended (string[]): array of recommended hashtags that could improve reach
               - reachPotential (string): textual description of the hashtags' potential reach
               - relevanceScore (number): 0-100 score of how relevant the hashtags are to the content
            
            The post ${includesMedia ? "includes" : "does not include"} media (image/video).
            Respond ONLY with the JSON object, no explanation or other text. Ensure the response follows this exact format with the exact property names specified.`
          },
          {
            role: "user",
            content: post
          }
        ],
        temperature: 0.5,
        max_tokens: 2048
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      console.error("API error:", errorData);
      throw new Error(`Error ${response.status}: ${errorData?.error?.message || 'Unknown error'}`);
    }

    const data = await response.json();
    const content = data.choices[0].message.content;
    
    try {
      const parsedContent = safeJsonParse(content);
      
      // Validate and provide defaults for critical properties
      const result: PostAnalysis = {
        score: parsedContent.score || parsedContent.overall_score || 0,
        engagementScore: parsedContent.engagementScore || parsedContent.engagement_score || 0,
        readabilityScore: parsedContent.readabilityScore || parsedContent.readability_score || 0,
        viralityScore: parsedContent.viralityScore || parsedContent.virality_score || 0,
        algorithmFriendly: Array.isArray(parsedContent.algorithmFriendly) ? 
          parsedContent.algorithmFriendly : 
          (Array.isArray(parsedContent.algorithm_friendly_features) ? 
            parsedContent.algorithm_friendly_features : []),
        optimizationOpportunities: Array.isArray(parsedContent.optimizationOpportunities) ? 
          parsedContent.optimizationOpportunities : 
          (Array.isArray(parsedContent.optimization_opportunities) ? 
            parsedContent.optimization_opportunities : []),
        optimizationTips: Array.isArray(parsedContent.optimizationTips) ? 
          parsedContent.optimizationTips : 
          (Array.isArray(parsedContent.optimization_tips) ? 
            parsedContent.optimization_tips : []),
        keyFactors: Array.isArray(parsedContent.keyFactors) ? 
          parsedContent.keyFactors : 
          (Array.isArray(parsedContent.key_algorithm_factors) ? 
            parsedContent.key_algorithm_factors : []),
        bestPostingTime: {
          daily: parsedContent.bestPostingTime?.daily || 
                parsedContent.best_posting_time_information?.daily || 
                parsedContent.best_posting_time?.daily_time || "",
          peakDays: parsedContent.bestPostingTime?.peakDays || 
                   parsedContent.best_posting_time_information?.peakDays ||
                   parsedContent.best_posting_time?.peak_days || ""
        },
        targetAudience: parsedContent.targetAudience || 
                       parsedContent.target_audience_description || 
                       parsedContent.target_audience || "",
        keywords: parsedContent.keywords || 
                 parsedContent.optimal_keywords_for_the_post || 
                 parsedContent.optimal_keywords || "",
        
        sentiment: {
          tone: parsedContent.sentiment?.tone || 
                parsedContent.sentiment_analysis?.tone || 
                "neutral",
          score: parsedContent.sentiment?.score || 
                parsedContent.sentiment_analysis?.score || 
                50,
          emotions: Array.isArray(parsedContent.sentiment?.emotions) ? 
                   parsedContent.sentiment.emotions : 
                   (Array.isArray(parsedContent.sentiment_analysis?.emotions) ?
                    parsedContent.sentiment_analysis.emotions : []),
          isPositive: parsedContent.sentiment?.isPositive || 
                     parsedContent.sentiment_analysis?.is_positive || 
                     false,
          isNegative: parsedContent.sentiment?.isNegative || 
                     parsedContent.sentiment_analysis?.is_negative || 
                     false,
          isNeutral: parsedContent.sentiment?.isNeutral || 
                    parsedContent.sentiment_analysis?.is_neutral || 
                    true
        },
        hashtags: {
          trending: Array.isArray(parsedContent.hashtags?.trending) ? 
                    parsedContent.hashtags.trending : 
                    (Array.isArray(parsedContent.hashtag_analysis?.trending) ? 
                     parsedContent.hashtag_analysis.trending : []),
          recommended: Array.isArray(parsedContent.hashtags?.recommended) ? 
                      parsedContent.hashtags.recommended : 
                      (Array.isArray(parsedContent.hashtag_analysis?.recommended) ? 
                       parsedContent.hashtag_analysis.recommended : []),
          reachPotential: parsedContent.hashtags?.reachPotential || 
                         parsedContent.hashtag_analysis?.reach_potential || 
                         "Low",
          relevanceScore: parsedContent.hashtags?.relevanceScore || 
                         parsedContent.hashtag_analysis?.relevance_score || 
                         0
        }
      };
      
      toast.success("Analysis complete!");
      return result;
      
    } catch (e) {
      console.error("Failed to parse JSON response:", content);
      throw new Error("Failed to parse analysis results");
    }
  } catch (error) {
    console.error("Analysis error:", error);
    toast.error(`Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    throw error;
  }
};

/**
 * Asynchronously rewrites a given post in a specified style and language using OpenAI API.
 */
export const rewritePost = async (
  post: string,
  style: string,
  limitLength: boolean,
  apiKey: string,
  model: OpenAIModel = "gpt-4o-mini",
  includeEmojis: boolean = true,
  multilingualOptions?: MultilingualOptions
): Promise<string> => {
  try {
    // Extract language settings if provided
    const languageSettings = multilingualOptions?.language 
      ? getLanguageSettings(multilingualOptions.language)
      : getLanguageSettings();
    
    // Determine language-specific character limit
    const characterLimit = limitLength 
      ? languageSettings.characterLimit || 280
      : undefined;
    
    // Create toast message with language info if specified
    const toastMessage = multilingualOptions?.language
      ? `Rewriting in ${style} style (${languageSettings.name})...`
      : `Rewriting in ${style} style...`;
    
    toast.info(toastMessage);
    
    // Determine if we need culturally adapted content
    const adaptForCulture = multilingualOptions?.adaptForCulture ?? false;
    const preserveHashtags = multilingualOptions?.preserveHashtags ?? true;
    const preserveMentions = multilingualOptions?.preserveMentions ?? true;
    
    // Extract hashtags and mentions to preserve if needed
    const hashtagRegex = /#([\w\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\u0870-\u089F\u08A0-\u08FF\u0900-\u097F\u0980-\u09FF\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]+)/g;
    const mentionRegex = /@([\w]+)/g;
    
    let hashtags: string[] = [];
    let mentions: string[] = [];
    
    if (preserveHashtags) {
      const hashtagMatches = [...post.matchAll(hashtagRegex)];
      hashtags = hashtagMatches.map(match => match[0]);
    }
    
    if (preserveMentions) {
      const mentionMatches = [...post.matchAll(mentionRegex)];
      mentions = mentionMatches.map(match => match[0]);
    }
    
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model,
        messages: [
          {
            role: "system",
            content: `You are an expert social media post writer. 
            Rewrite the provided post in a ${style} style ${languageSettings.code !== 'en' ? `in ${languageSettings.name}` : ''}. 
            ${characterLimit ? `Limit the response to ${characterLimit} characters maximum.` : ""}
            ${!includeEmojis ? "Do not include any emojis in your response." : "Feel free to include relevant emojis for engagement."}
            ${adaptForCulture ? `Adapt the content to be culturally relevant for ${languageSettings.name} speakers.` : ""}
            ${preserveHashtags && hashtags.length > 0 ? `Include these hashtags if relevant: ${hashtags.join(', ')}` : ""}
            ${preserveMentions && mentions.length > 0 ? `Preserve these mentions: ${mentions.join(', ')}` : ""}
            Your response should be ONLY the rewritten post, with no explanations or other text.`
          },
          {
            role: "user",
            content: post
          }
        ],
        temperature: 0.7,
        max_tokens: characterLimit ? Math.ceil(characterLimit / 2) : 280
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      throw new Error(`Error ${response.status}: ${errorData?.error?.message || 'Unknown error'}`);
    }

    const data = await response.json();
    let rewrittenPost = data.choices[0].message.content.trim();
    
    // Post-processing for language-specific requirements
    if (multilingualOptions?.language) {
      const language = getLanguageSettings(multilingualOptions.language);
      
      // Handle character limits for CJK languages (they count double)
      if (language.characterLimit && language.characterLimit < 280 && limitLength) {
        if (rewrittenPost.length > language.characterLimit) {
          rewrittenPost = rewrittenPost.substring(0, language.characterLimit);
        }
      }
      
      // Add RTL markers for right-to-left languages if needed
      if (language.rtl) {
        // Unicode RLM (Right-to-Left Mark) character
        rewrittenPost = '\u200F' + rewrittenPost;
      }
    }
    
    toast.success("Rewrite complete!");
    return rewrittenPost;
  } catch (error) {
    console.error("Rewrite error:", error);
    toast.error(`Rewrite failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    throw error;
  }
};

/**
 * Calculates engagement metrics based on likes, reposts, replies, followers, and optional impressions.
 */
export const calculateEngagementRate = (
  likes: number,
  reposts: number,
  replies: number,
  followers: number,
  impressions?: number
): EngagementMetrics => {
  // If impressions are provided and valid, use them instead of followers
  const denominator = (impressions && impressions > 0) ? impressions : followers;
  
  // Avoid division by zero
  if (denominator === 0) {
    return {
      engagementRate: 0,
      likeRate: 0,
      repostRate: 0,
      replyRate: 0,
      performanceText: "No followers or impressions data available"
    };
  }
  
  // Calculate rates
  const likeRate = parseFloat(((likes / denominator) * 100).toFixed(2));
  const repostRate = parseFloat(((reposts / denominator) * 100).toFixed(2));
  const replyRate = parseFloat(((replies / denominator) * 100).toFixed(2));
  
  // Total engagement rate
  const engagementRate = parseFloat(((likes + reposts + replies) / denominator * 100).toFixed(2));
  
  // Generate performance text
  let performanceText = "";
  
  if (impressions) {
    if (engagementRate > 3) performanceText = "Excellent engagement rate based on impressions! This post is performing well.";
    else if (engagementRate > 1.5) performanceText = "Good engagement rate based on impressions. This post is performing above average.";
    else if (engagementRate > 0.8) performanceText = "Average engagement rate based on impressions.";
    else performanceText = "Below average engagement rate based on impressions. Consider optimizing future posts.";
  } else {
    if (engagementRate > 1) performanceText = "Excellent engagement rate! This post is performing well.";
    else if (engagementRate > 0.5) performanceText = "Good engagement rate. This post is performing above average.";
    else if (engagementRate > 0.2) performanceText = "Average engagement rate for X (Twitter).";
    else performanceText = "Below average engagement rate. Consider optimizing future posts.";
  }
  
  return {
    engagementRate,
    likeRate,
    repostRate,
    replyRate,
    performanceText
  };
};

/**
 * Detects the language of a given text based on character ranges.
 */
export const detectLanguage = (text: string): string => {
  // This is a simplified detection based on character ranges
  // For a real implementation, use a proper language detection library
  
  // Check for CJK characters (Chinese, Japanese, Korean)
  const hasCJK = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f\u3131-\u318e\uac00-\ud7a3]/.test(text);
  if (hasCJK) {
    // Further differentiate between Chinese, Japanese and Korean
    // This is very simplified - a real implementation would be more sophisticated
    const hasJapanese = /[\u3040-\u309f\u30a0-\u30ff]/.test(text);
    if (hasJapanese) return "ja";
    
    const hasKorean = /[\u3131-\u318e\uac00-\ud7a3]/.test(text);
    if (hasKorean) return "ko";
    
    return "zh"; // Default to Chinese if other CJK not detected
  }
  
  // Check for Arabic script
  if (/[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufefc]/.test(text)) {
    return "ar";
  }
  
  // Check for Cyrillic (Russian)
  if (/[\u0400-\u04FF]/.test(text)) {
    return "ru";
  }
  
  // Check for Devanagari (Hindi)
  if (/[\u0900-\u097F]/.test(text)) {
    return "hi";
  }
  
  // Check for Spanish/Portuguese specific characters
  if (/[áéíóúüñÁÉÍÓÚÜÑ]/.test(text)) {
    // This is a very rough distinction and would need refinement
    if (/[çãõÃÕ]/.test(text)) {
      return "pt"; // More likely Portuguese if it has these
    }
    return "es"; // Default to Spanish
  }
  
  // Check for French specific characters
  if (/[àâæçéèêëîïôœùûüÿÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ]/.test(text)) {
    return "fr";
  }
  
  // Check for German specific characters
  if (/[äöüßÄÖÜ]/.test(text)) {
    return "de";
  }
  
  // Check for Italian specific patterns
  if (/[àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ]/.test(text) && /\b(di|il|la|che|per|con)\b/i.test(text)) {
    return "it";
  }
  
  // Default to English for Latin script and others
  return "en";
};

/**
 * Validates text against a specified language's character limit, considering CJK character counting rules.
 */
export const validateTextForLanguage = (text: string, languageCode: string, enforceLimit: boolean = true): { 
  valid: boolean; 
  remainingChars: number; 
  message?: string 
} => {
  const language = getLanguageSettings(languageCode);
  const limit = language.characterLimit || 280;
  
  // For CJK languages, some platforms count each character as 2
  const effectiveLength = language.code === 'ja' || language.code === 'ko' || language.code === 'zh'
    ? [...text].reduce((acc, char) => {
        // Check if character is CJK
        if (/[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f\u3131-\u318e\uac00-\ud7a3]/.test(char)) {
          return acc + 2;
        }
        return acc + 1;
      }, 0)
    : text.length;
  
  const remainingChars = limit - effectiveLength;
  const valid = remainingChars >= 0;
  
  if (!valid && enforceLimit) {
    return {
      valid: false,
      remainingChars,
      message: `Text exceeds the ${limit} character limit for ${language.name} by ${Math.abs(remainingChars)} ${Math.abs(remainingChars) === 1 ? 'character' : 'characters'}`
    };
  }
  
  return {
    valid: true,
    remainingChars
  };
};
