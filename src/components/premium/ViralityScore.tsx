import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { TrendingUp, Zap, CheckCircle, AlertCircle } from "lucide-react";
import { ScoreBar } from "../analysis/ScoreBar";

interface ViralityScoreProps {
  includeEmojis?: boolean;
}

export function ViralityScore({ includeEmojis = true }: ViralityScoreProps) {
  // Emoji or text representations based on toggle
  const viralIcon = includeEmojis ? "🚀 " : "";
  const statsIcon = includeEmojis ? "📊 " : "";
  const positiveIcon = includeEmojis ? "✅ " : "";
  const negativeIcon = includeEmojis ? "⚠️ " : "";
  
  // Mock data for virality predictions
  const viralScore = 68;
  const engagementVelocity = 72;
  const contentRelevance = 85;
  const structureScore = 63;
  
  const positiveFactors = [
    "Concise messaging with clear value proposition",
    "Includes trending topic relevant to your audience",
    "Good use of emotional triggers with clear call-to-action"
  ];
  
  const negativeFactors = [
    "Post timing could be optimized for better reach",
    "Consider adding a question to boost engagement",
    "Post lacks visual elements that typically increase virality"
  ];
    
  return (
    <Card className="bg-secondary border-gray-700">
      <CardHeader className="flex flex-row items-center justify-between pb-2 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <TrendingUp size={18} className="text-green-400" />
          <h3 className="text-lg font-semibold">{viralIcon}Virality Score</h3>
        </div>
        <span className="px-2 py-1 bg-blue-900/40 text-blue-400 text-xs border border-blue-700/50 rounded-full">
          PREMIUM
        </span>
      </CardHeader>
      <CardContent className="pt-4">
        <div className="space-y-4">
          <div className="p-4 bg-gray-800/50 rounded-md border border-gray-700">
            <div className="flex items-center justify-center mb-4">
              <div className="relative w-28 h-28">
                <div className="absolute inset-0 flex items-center justify-center">
                  <Zap size={40} className={viralScore >= 70 ? "text-yellow-400" : "text-blue-400"} />
                </div>
                <svg className="w-28 h-28 transform -rotate-90">
                  <circle
                    cx="56"
                    cy="56"
                    r="50"
                    fill="transparent"
                    stroke="currentColor"
                    strokeWidth="10"
                    strokeDasharray="314.16"
                    strokeDashoffset={(1 - viralScore / 100) * 314.16}
                    className={`transition-all duration-1000 ease-out ${
                      viralScore < 40 ? "text-red-500" :
                      viralScore < 70 ? "text-yellow-500" :
                      "text-green-500"
                    }`}
                  />
                  <circle
                    cx="56"
                    cy="56"
                    r="50"
                    fill="transparent"
                    stroke="currentColor"
                    strokeWidth="10"
                    className="text-gray-700"
                  />
                </svg>
              </div>
            </div>
            
            <div className="text-center mb-4">
              <h4 className="font-semibold text-xl">{viralScore}% Viral Potential</h4>
              <p className="text-sm text-gray-400">
                {viralScore >= 80 ? "Extremely high viral potential" : 
                 viralScore >= 70 ? "High viral potential" :
                 viralScore >= 50 ? "Moderate viral potential" :
                 "Low viral potential"}
              </p>
            </div>
          </div>
          
          <div className="p-4 bg-gray-800/50 rounded-md border border-gray-700">
            <h4 className="font-medium mb-3 flex items-center gap-2">
              <TrendingUp size={16} className="text-green-500" />
              {statsIcon}Virality Factors
            </h4>
            
            <div className="space-y-3">
              <ScoreBar label="Engagement Velocity" score={engagementVelocity} />
              <ScoreBar label="Content Relevance" score={contentRelevance} />
              <ScoreBar label="Structure & Format" score={structureScore} />
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-3 bg-green-900/20 rounded-md border border-green-700/50">
              <h4 className="font-medium mb-2 flex items-center gap-2">
                <CheckCircle size={16} className="text-green-500" />
                {positiveIcon}Positive Factors
              </h4>
              <ul className="space-y-2 text-sm">
                {positiveFactors.map((factor, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-green-500 mt-1">•</span>
                    <span>{factor}</span>
                  </li>
                ))}
              </ul>
            </div>
            
            <div className="p-3 bg-red-900/20 rounded-md border border-red-700/50">
              <h4 className="font-medium mb-2 flex items-center gap-2">
                <AlertCircle size={16} className="text-red-500" />
                {negativeIcon}Areas to Improve
              </h4>
              <ul className="space-y-2 text-sm">
                {negativeFactors.map((factor, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-red-500 mt-1">•</span>
                    <span>{factor}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
          
          <div className="p-3 bg-gray-900/30 rounded-md border border-yellow-700/50 text-center">
            <p className="text-sm text-yellow-400">
              Upgrade to Premium for detailed virality predictions and trend compatibility analysis
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
