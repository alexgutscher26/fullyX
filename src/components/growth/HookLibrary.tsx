import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Copy, Wand2, X, Search } from "lucide-react";
import { toast } from "@/components/ui/sonner";
import { generateHook } from "@/services/ai-hooks";
import { useState } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface HookLibraryProps {
  includeEmojis: boolean;
}

export function HookLibrary({ includeEmojis }: HookLibraryProps) {
  const hookIcon = includeEmojis ? "🎣 " : "";
  const [topic, setTopic] = useState("");
  const [generating, setGenerating] = useState(false);
  const [previewHook, setPreviewHook] = useState<string | null>(null);
  const [previewCategory, setPreviewCategory] = useState<string | null>(null);

  // Sample hook categories and examples
  const hookCategories = [
    {
      name: "Curiosity",
      emote: "🤔",
      hooks: [
        "You won't believe what happened when...",
        "I discovered the secret to... and it's not what you think",
        "The surprising truth about...",
        "Most people get this wrong about...",
      ],
    },
    {
      name: "Value-Based",
      emote: "💎",
      hooks: [
        "3 simple techniques to increase your...",
        "The framework I used to grow from 0 to...",
        "How to [desired outcome] in just [timeframe]",
        "The exact system that helped me...",
      ],
    },
    {
      name: "Problem-Solution",
      emote: "🔧",
      hooks: [
        "Struggling with [problem]? Here's how to fix it:",
        "If you're facing [challenge], try this:",
        "The biggest mistake people make with [topic]",
        "How I solved [problem] after years of struggle",
      ],
    },
    {
      name: "Social Proof",
      emote: "👥",
      hooks: [
        "How my client went from [before] to [after]...",
        "Case study: $X in [timeframe] using this approach",
        "I helped 100+ people achieve [result]. Here's how:",
        "This strategy helped me get featured in...",
      ],
    },
  ];

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Hook copied to clipboard!");
  };

  const handleGenerateHook = async (category: string) => {
    if (generating) return;

    setGenerating(true);
    try {
      const newHook = await generateHook({ category, topic });
      if (newHook) {
        setPreviewHook(newHook);
        setPreviewCategory(category);
      }
    } catch (error) {
      console.error("Error generating hook:", error);
      toast.error("Failed to generate hook");
    } finally {
      setGenerating(false);
    }
  };

  return (
    <>
      <Dialog open={previewHook !== null} onOpenChange={() => setPreviewHook(null)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {previewCategory && includeEmojis && (
                <span>
                  {hookCategories.find(c => c.name === previewCategory)?.emote}
                </span>
              )}
              AI-Generated Hook Preview
            </DialogTitle>
          </DialogHeader>
          <div className="p-4 bg-gray-800/50 rounded-md border border-gray-700">
            <p className="text-sm mb-4">{previewHook}</p>
            <div className="flex justify-end gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPreviewHook(null)}
              >
                <X size={14} className="mr-2" />
                Close
              </Button>
              <Button
                size="sm"
                onClick={() => {
                  if (previewHook) {
                    copyToClipboard(previewHook);
                    setPreviewHook(null);
                  }
                }}
              >
                <Copy size={14} className="mr-2" />
                Copy to Clipboard
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <Card className="bg-secondary border-gray-700">
        <CardHeader className="border-b border-gray-700 pb-2">
          <h3 className="text-lg font-semibold flex items-center gap-1">
            {hookIcon}Hook Library
          </h3>
          <p className="text-sm text-muted-foreground">
            Proven hooks for increasing engagement
          </p>
        </CardHeader>
        <CardContent className="pt-4">
          {/* Topic search input */}
          <div className="mb-4">
            <div className="flex gap-2">
              <div className="relative flex-grow">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <input
                  type="text"
                  placeholder="Enter a topic to customize hooks..."
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  className="w-full pl-8 h-9 rounded-md border border-gray-700 bg-gray-800/50 px-3 py-1 text-sm"
                />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {hookCategories.map((category) => (
              <div key={category.name} className="space-y-2">
                <h4 className="font-medium flex items-center gap-2">
                  {includeEmojis && <span>{category.emote}</span>} {category.name}
                </h4>
                <div className="space-y-2">
                  {category.hooks.map((hook, index) => (
                    <div
                      key={index}
                      className="p-2 bg-gray-800/50 rounded-md border border-gray-700 flex justify-between items-center gap-2"
                    >
                      <p className="text-sm">{hook}</p>
                      <div className="flex gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-8 w-8 p-0"
                          onClick={() => copyToClipboard(hook)}
                        >
                          <Copy size={14} />
                          <span className="sr-only">Copy</span>
                        </Button>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-8 w-8 p-0"
                              onClick={() => handleGenerateHook(category.name)}
                              disabled={generating}
                            >
                              <Wand2
                                size={14}
                                className={generating ? "animate-pulse" : ""}
                              />
                              <span className="sr-only">Generate AI Hook</span>
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Generate AI-enhanced hook for this category</p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </>
  );
}