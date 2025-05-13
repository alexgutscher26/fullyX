import { useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Sparkles, Type, Megaphone, Target } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface AIRewriterProps {
  post: string;
  includeEmojis?: boolean;
}

export function AIRewriter({ post, includeEmojis = true }: AIRewriterProps) {
  const [selectedStyle, setSelectedStyle] = useState("viral");
  
  // Emoji or text representations based on toggle
  const magicIcon = includeEmojis ? "✨ " : "";
  const styleIcon = includeEmojis ? "🎭 " : "";
  const hookIcon = includeEmojis ? "🎣 " : "";
  const targetIcon = includeEmojis ? "🎯 " : "";
  
  // Mock data for AI rewriter suggestions
  const hookSuggestions = [
    "Did you know that 82% of top creators...",
    "Here's the uncomfortable truth about...",
    "I tested 5 different approaches and found...",
    "The secret to more engagement isn't what you think..."
  ];
  
  const ctaSuggestions = [
    "What's your experience with this? Comment below!",
    "Reply with '1' if you want more tips like this",
    "Tag someone who needs to see this",
    "Save this for later when you need it most"
  ];
    
  return (
    <Card className="bg-secondary border-gray-700">
      <CardHeader className="flex flex-row items-center justify-between pb-2 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <Sparkles size={18} className="text-purple-400" />
          <h3 className="text-lg font-semibold">{magicIcon}AI Post Optimizer</h3>
        </div>
        <span className="px-2 py-1 bg-blue-900/40 text-blue-400 text-xs border border-blue-700/50 rounded-full">
          PREMIUM
        </span>
      </CardHeader>
      <CardContent className="pt-4">
        <div className="space-y-4">
          <div className="p-4 bg-gray-800/50 rounded-md border border-gray-700">
            <h4 className="font-medium mb-3 flex items-center gap-2">
              <Type size={16} className="text-blue-500" />
              {styleIcon}Rewrite Styles
            </h4>
            
            <Tabs defaultValue={selectedStyle} className="w-full" onValueChange={setSelectedStyle}>
              <TabsList className="w-full grid grid-cols-5 mb-4 bg-gray-900/50">
                <TabsTrigger value="viral">Viral</TabsTrigger>
                <TabsTrigger value="professional">Professional</TabsTrigger>
                <TabsTrigger value="controversial">Controversial</TabsTrigger>
                <TabsTrigger value="storytelling">Storytelling</TabsTrigger>
                <TabsTrigger value="educational">Educational</TabsTrigger>
              </TabsList>
            </Tabs>
            
            <div className="p-3 rounded-md bg-blue-900/20 border border-blue-700/40">
              <p className="text-sm text-blue-300 mb-2">Original post:</p>
              <p className="text-sm">{post || "Your post will appear here"}</p>
            </div>
            
            <div className="p-3 mt-3 rounded-md bg-purple-900/20 border border-purple-700/40">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm text-purple-300">Optimized {selectedStyle} version:</p>
                <Button disabled size="sm" variant="ghost" className="text-xs h-7 px-2">
                  <Sparkles size={14} className="mr-1" /> 
                  Optimize
                </Button>
              </div>
              <p className="text-sm text-gray-400 italic">Upgrade to Premium to access AI post optimization</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-3 bg-gray-800/50 rounded-md border border-gray-700">
              <h4 className="font-medium mb-2 flex items-center gap-2">
                <Megaphone size={16} className="text-green-500" />
                {hookIcon}Suggested Hooks
              </h4>
              <ul className="space-y-2 text-sm">
                {hookSuggestions.map((hook, index) => (
                  <li key={index} className="p-2 bg-gray-900/60 rounded flex items-start gap-2 border border-gray-700/60">
                    <span className="text-green-500 mt-1">•</span>
                    <span>{hook}</span>
                  </li>
                ))}
              </ul>
            </div>
            
            <div className="p-3 bg-gray-800/50 rounded-md border border-gray-700">
              <h4 className="font-medium mb-2 flex items-center gap-2">
                <Target size={16} className="text-blue-500" />
                {targetIcon}Call-to-Actions
              </h4>
              <ul className="space-y-2 text-sm">
                {ctaSuggestions.map((cta, index) => (
                  <li key={index} className="p-2 bg-gray-900/60 rounded flex items-start gap-2 border border-gray-700/60">
                    <span className="text-blue-500 mt-1">•</span>
                    <span>{cta}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
          
          <div className="p-3 bg-gray-900/30 rounded-md border border-yellow-700/50 text-center">
            <p className="text-sm text-yellow-400">
              Upgrade to Premium to access AI-powered post rewriting and optimization
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
