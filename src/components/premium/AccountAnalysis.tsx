import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { BarChart, LineChart, PieChart, Users } from "lucide-react";
import { ScoreBar } from "../analysis/ScoreBar";

interface AccountAnalysisProps {
  includeEmojis?: boolean;
}

export function AccountAnalysis({ includeEmojis = true }: AccountAnalysisProps) {
  // Emoji or text representations based on toggle
  const analysisIcon = includeEmojis ? "📊 " : "";
  const growthIcon = includeEmojis ? "📈 " : "";
  const contentIcon = includeEmojis ? "📝 " : "";
  const performanceIcon = includeEmojis ? "🏆 " : "";
  
  // Mock data for account analysis
  const contentTypes = [
    { type: "Text Only", percentage: 45 },
    { type: "Image", percentage: 35 },
    { type: "Video", percentage: 15 },
    { type: "Link", percentage: 5 }
  ];
  
  const engagementStats = {
    avgLikes: 128,
    avgReplies: 24,
    avgReposts: 36,
    growthRate: "+8.5%"
  };
    
  return (
    <Card className="bg-secondary border-gray-700">
      <CardHeader className="flex flex-row items-center justify-between pb-2 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <Users size={18} className="text-green-400" />
          <h3 className="text-lg font-semibold">{analysisIcon}Account Analysis</h3>
        </div>
        <span className="px-2 py-1 bg-blue-900/40 text-blue-400 text-xs border border-blue-700/50 rounded-full">
          PREMIUM
        </span>
      </CardHeader>
      <CardContent className="pt-4">
        <div className="space-y-4">
          <div className="p-4 bg-gray-800/50 rounded-md border border-gray-700">
            <h4 className="font-medium mb-3 flex items-center gap-2">
              <LineChart size={16} className="text-blue-500" />
              {growthIcon}Growth Trends
            </h4>
            
            <div className="h-40 bg-gray-900/80 rounded-md border border-gray-700 flex items-center justify-center">
              <div className="text-center">
                <p className="text-gray-400 text-sm">Follower Growth Rate</p>
                <p className="text-2xl font-bold text-green-500">{engagementStats.growthRate}</p>
                <p className="text-gray-500 text-xs">last 30 days</p>
                <p className="mt-2 text-sm text-yellow-400">Upgrade to see complete growth analytics</p>
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-3 bg-gray-800/50 rounded-md border border-gray-700">
              <h4 className="font-medium mb-2 flex items-center gap-2">
                <PieChart size={16} className="text-purple-500" />
                {contentIcon}Content Breakdown
              </h4>
              <div className="space-y-3">
                {contentTypes.map((content) => (
                  <div key={content.type} className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>{content.type}</span>
                      <span>{content.percentage}%</span>
                    </div>
                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-purple-600 rounded-full"
                        style={{ width: `${content.percentage}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="p-3 bg-gray-800/50 rounded-md border border-gray-700">
              <h4 className="font-medium mb-2 flex items-center gap-2">
                <BarChart size={16} className="text-green-500" />
                {performanceIcon}Average Performance
              </h4>
              
              <div className="space-y-3 mt-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Average Likes:</span>
                  <span className="font-medium">{engagementStats.avgLikes}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Average Replies:</span>
                  <span className="font-medium">{engagementStats.avgReplies}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Average Reposts:</span>
                  <span className="font-medium">{engagementStats.avgReposts}</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="p-3 bg-gray-900/30 rounded-md border border-yellow-700/50 text-center">
            <p className="text-sm text-yellow-400">
              Upgrade to Premium for complete account analytics and personalized growth insights
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
