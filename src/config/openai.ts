import OpenAI from 'openai';

// Initialize OpenAI client with validation
let apiKey: string | undefined;

try {
  // In browser environment, try to get from local storage first
  if (typeof window !== 'undefined') {
    const storedKey = localStorage.getItem('x_post_roast_api_key');
    if (storedKey) {
      apiKey = storedKey;
    }
    // Fallback to Next.js data if available
    if (!apiKey) {
      apiKey = window.__NEXT_DATA__?.props?.pageProps?.apiKey;
    }
  }
} catch (error) {
  console.error('Error initializing API key:', error);
}

export const openai = new OpenAI({
  apiKey: apiKey || '',
  dangerouslyAllowBrowser: true
});

// Hook generation prompt template
export const HOOK_GENERATION_PROMPT = {
  system: `You are a professional copywriter specializing in creating engaging hooks and headlines. 
  Your task is to generate a compelling hook based on the given category and topic.
  The hook should be attention-grabbing, relevant, and follow the category's style.
  
  Categories and their characteristics:
  - Curiosity: Create intrigue and mystery, make readers want to learn more
  - Value-Based: Focus on concrete benefits and results
  - Problem-Solution: Address pain points and offer clear solutions
  - Social Proof: Leverage credibility and real-world success
  
  Keep the tone professional yet conversational. Ensure the hook is concise and impactful.`,
  
  user: (category: string, topic: string, tone: string) => 
    `Generate a compelling hook for the following:
    Category: ${category}
    Topic: ${topic}
    Tone: ${tone}
    
    The hook should be a single sentence or phrase that captures attention and drives engagement.`
};