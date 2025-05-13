import { toast } from "@/components/ui/sonner";
import { openai, HOOK_GENERATION_PROMPT } from "@/config/openai";

interface HookGenerationParams {
  category: string;
  topic?: string;
  tone?: string;
}

export async function generateHook({ category, topic = '', tone = 'professional' }: HookGenerationParams): Promise<string> {
  if (!category) {
    toast.error('Category is required for hook generation');
    return '';
  }

  try {
    // Validate OpenAI configuration
    if (!openai.apiKey) {
      toast.error('OpenAI API key is not configured');
      return '';
    }

    // Call OpenAI API to generate a hook
    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [
        { role: "system", content: HOOK_GENERATION_PROMPT.system },
        { role: "user", content: HOOK_GENERATION_PROMPT.user(category, topic, tone) }
      ],
      temperature: 0.7,
      max_tokens: 100,
      presence_penalty: 0.3,
      frequency_penalty: 0.3
    });

    // Extract the generated hook from the response
    const generatedHook = completion.choices[0]?.message?.content?.trim();

    if (!generatedHook) {
      throw new Error('No hook was generated');
    }

    return generatedHook;

  } catch (error) {
    console.error('Error generating hook:', error);
    
    // Handle specific API errors
    if (error instanceof Error) {
      if (error.message.includes('API key') || error.message.includes('auth')) {
        toast.error('Invalid or missing API key. Please check your configuration.');
      } else if (error.message.includes('quota') || error.message.includes('rate_limit')) {
        toast.error('API quota exceeded. Please try again later.');
      } else if (error.message.includes('timeout') || error.message.includes('network')) {
        toast.error('Network error. Please check your connection.');
      } else {
        toast.error('Failed to generate hook. Please try again.');
      }
    } else {
      toast.error('An unexpected error occurred');
    }

    return '';
  }
}