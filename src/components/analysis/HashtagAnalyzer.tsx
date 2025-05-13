
import React from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Hash, TrendingUp } from "lucide-react";
import { ScoreBar } from "./ScoreBar";

interface HashtagAnalyzerProps {
  hashtags: {
    trending: string[];
    recommended: string[];
    reachPotential: string;
    relevanceScore: number;
  };
  includeEmojis?: boolean;
}

export function HashtagAnalyzer({ hashtags, includeEmojis = true }: HashtagAnalyzerProps) {
  // Set defaults if hashtags is undefined
  const {
    trending = [],
    recommended = [],
    reachPotential = "Low",
    relevanceScore = 0
  } = hashtags || {};

  // Choose emoji or text representation based on toggle
  const trendingIcon = includeEmojis ? "🔥 " : "";
  const recommendedIcon = includeEmojis ? "✨ " : "";
  const reachIcon = includeEmojis ? "📊 " : "";
  const tipIcon = includeEmojis ? "💡 " : "";

  return (
    <Card className="bg-secondary border-gray-700">
      <CardHeader className="flex flex-row items-center justify-between pb-2 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <Hash size={18} className="text-purple-400" />
          <h3 className="text-lg font-semibold">Hashtag Analysis</h3>
        </div>
      </CardHeader>
      <CardContent className="pt-4">
        <div className="space-y-4">
          <ScoreBar label="Hashtag Relevance" score={relevanceScore} />
          
          <div className="p-3 bg-gray-800/50 rounded-md border border-gray-700">
            <h4 className="font-medium mb-2 flex items-center gap-2">
              <TrendingUp size={16} className="text-green-500" />
              {trendingIcon}Trending Hashtags
            </h4>
            <div className="flex flex-wrap gap-2">
              {trending.length > 0 ? (
                trending.map((tag, index) => (
                  <span
                    key={index}
                    className="px-3 py-1 bg-green-900/40 text-green-400 border border-green-700/50 rounded-full text-sm animate-pulse"
                    style={{ animationDelay: `${index * 200}ms` }}
                  >
                    #{tag}
                  </span>
                ))
              ) : (
                <p className="text-sm text-gray-400">No trending hashtags found in your post</p>
              )}
            </div>
          </div>
          
          <div className="p-3 bg-gray-800/50 rounded-md border border-gray-700">
            <h4 className="font-medium mb-2">{recommendedIcon}Recommended Hashtags</h4>
            <div className="flex flex-wrap gap-2">
              {recommended.length > 0 ? (
                recommended.map((tag, index) => (
                  <span
                    key={index}
                    className="px-3 py-1 bg-blue-900/40 text-blue-400 border border-blue-700/50 rounded-full text-sm"
                  >
                    #{tag}
                  </span>
                ))
              ) : (
                <p className="text-sm text-gray-400">No recommendations available</p>
              )}
            </div>
          </div>
          
          <div className="p-3 bg-gray-800/50 rounded-md border border-gray-700">
            <h4 className="font-medium mb-1">{reachIcon}Reach Potential</h4>
            <p className="text-sm">{reachPotential}</p>
          </div>
          
          <div className="p-3 bg-gray-800/50 rounded-md border border-gray-700">
            <p className="text-sm text-gray-300">
              <strong>{tipIcon}Tip:</strong> For better reach, try to include 1-2 trending hashtags and 1-3 niche hashtags related to your content.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}