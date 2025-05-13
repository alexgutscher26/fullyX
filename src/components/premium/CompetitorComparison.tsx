import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Users, ArrowRightLeft, Gauge } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ScoreBar } from "../analysis/ScoreBar";

interface CompetitorComparisonProps {
  includeEmojis?: boolean;
}

export function CompetitorComparison({ includeEmojis = true }: CompetitorComparisonProps) {
  // Emoji or text representations based on toggle
  const comparisonIcon = includeEmojis ? "🔍 " : "";
  const versusIcon = includeEmojis ? "⚔️ " : "";
  const statsIcon = includeEmojis ? "📊 " : "";
  
  return (
    <Card className="bg-secondary border-gray-700">
      <CardHeader className="flex flex-row items-center justify-between pb-2 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <ArrowRightLeft size={18} className="text-orange-400" />
          <h3 className="text-lg font-semibold">{comparisonIcon}Competitor Comparison</h3>
        </div>
        <span className="px-2 py-1 bg-blue-900/40 text-blue-400 text-xs border border-blue-700/50 rounded-full">
          PREMIUM
        </span>
      </CardHeader>
      <CardContent className="pt-4">
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm text-gray-400 mb-1 block">Your Handle</label>
              <Input placeholder="@yourhandle" className="bg-gray-900/60 border-gray-700" />
            </div>
            <div>
              <label className="text-sm text-gray-400 mb-1 block">Competitor Handle</label>
              <Input placeholder="@competitor" className="bg-gray-900/60 border-gray-700" />
            </div>
          </div>
          
          <div className="flex justify-center">
            <Button disabled variant="outline" className="border-gray-700 bg-gray-800 text-gray-300">
              {versusIcon}Compare Accounts
            </Button>
          </div>
          
          <div className="p-4 bg-gray-800/50 rounded-md border border-gray-700">
            <h4 className="font-medium mb-3 flex items-center justify-center gap-2">
              <Gauge size={16} className="text-blue-500" />
              {statsIcon}Comparative Metrics Preview
            </h4>
            
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-3">
                <p className="text-sm text-center text-blue-400">Your Account</p>
                <ScoreBar label="Engagement Rate" score={65} />
                <ScoreBar label="Growth Rate" score={42} />
                <ScoreBar label="Content Quality" score={78} />
                <ScoreBar label="Posting Consistency" score={85} />
              </div>
              
              <div className="space-y-3">
                <p className="text-sm text-center text-orange-400">Competitor</p>
                <ScoreBar label="Engagement Rate" score={72} />
                <ScoreBar label="Growth Rate" score={58} />
                <ScoreBar label="Content Quality" score={66} />
                <ScoreBar label="Posting Consistency" score={92} />
              </div>
            </div>
          </div>
          
          <div className="p-3 bg-gray-900/30 rounded-md border border-yellow-700/50 text-center">
            <p className="text-sm text-yellow-400">
              Upgrade to Premium to access detailed competitor analytics and benchmarking
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
