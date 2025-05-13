import { toast } from "@/components/ui/sonner";
import { openai } from "@/config/openai";

interface ABTestingAnalysisParams {
  versionA: string;
  versionB: string;
}

interface ABTestingResult {
  winnerVersion: "A" | "B";
  scoreA: number;
  scoreB: number;
  reasons: string[];
}

const AB_TESTING_PROMPT = {
  system: `You are an expert content analyst specializing in social media performance prediction.
  Your task is to analyze two versions of content and predict which one will perform better based on:
  - Hook effectiveness and curiosity generation
  - Clarity and value proposition
  - Engagement potential and emotional resonance
  - Call to action strength
  
  For each version, provide a score between 60-100 and explain the reasoning.
  Focus on actionable insights and specific elements that make one version stronger than the other.`,
  
  user: (versionA: string, versionB: string) =>
    `Analyze these two content versions and predict their performance:

    Version A:
    ${versionA}

    Version B:
    ${versionB}

    Provide the analysis in the following JSON format:
    {
      "winnerVersion": "A" or "B",
      "scoreA": number between 60-100,
      "scoreB": number between 60-100,
      "reasons": [three specific reasons why the winning version is better]
    }`
};

export async function analyzeABTest({ versionA, versionB }: ABTestingAnalysisParams): Promise<ABTestingResult> {
  if (!versionA || !versionB) {
    toast.error('Both versions are required for analysis');
    throw new Error('Both versions are required');
  }

  try {
    if (!openai.apiKey) {
      toast.error('OpenAI API key is not configured');
      throw new Error('OpenAI API key not configured');
    }

    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [
        { role: "system", content: AB_TESTING_PROMPT.system + "\nPlease provide your response in valid JSON format." },
        { role: "user", content: AB_TESTING_PROMPT.user(versionA, versionB) }
      ],
      temperature: 0.7,
      max_tokens: 500
    });

    const resultContent = completion.choices[0]?.message?.content;
    if (!resultContent) {
      throw new Error('No analysis was generated');
    }

    const result = JSON.parse(resultContent) as ABTestingResult;
    return result;

  } catch (error) {
    console.error('Error analyzing A/B test:', error);
    if (error instanceof Error) {
      if (error.message.includes('API key') || error.message.includes('auth')) {
        toast.error('API authentication failed. Please check your API key.');
      } else {
        toast.error('Failed to analyze versions. Please try again.');
      }
    }
    throw error;
  }
}