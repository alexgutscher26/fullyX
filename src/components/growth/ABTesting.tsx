
import { useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ArrowRight, Scale } from "lucide-react";
import { toast } from "@/components/ui/sonner";
import { analyzeABTest } from "@/services/ab-testing";

interface ABTestingProps {
  includeEmojis: boolean;
}

export function ABTesting({ includeEmojis }: ABTestingProps) {
  const [versionA, setVersionA] = useState("");
  const [versionB, setVersionB] = useState("");
  const [results, setResults] = useState<null | {
    winnerVersion: "A" | "B";
    scoreA: number;
    scoreB: number;
    reasons: string[];
  }>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  const abIcon = includeEmojis ? "🧪 " : "";
  
  const handleCompare = async () => {
    if (!versionA.trim() || !versionB.trim()) {
      toast.error("Please enter both versions to compare");
      return;
    }
    
    setIsAnalyzing(true);
    
    try {
      const analysis = await analyzeABTest({
        versionA: versionA.trim(),
        versionB: versionB.trim()
      });
      
      setResults(analysis);
      toast.success("Analysis complete!");
    } catch (error) {
      // Error is already handled in the service
      setResults(null);
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  return (
    <Card className="bg-secondary border-gray-700">
      <CardHeader className="border-b border-gray-700 pb-2">
        <h3 className="text-lg font-semibold flex items-center gap-1">
          {abIcon}A/B Testing
        </h3>
        <p className="text-sm text-muted-foreground">
          Compare two versions to predict better performance
        </p>
      </CardHeader>
      <CardContent className="pt-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div className="space-y-2">
            <h4 className="font-medium">Version A</h4>
            <Textarea 
              placeholder="Enter first version of your post..."
              className="h-36 bg-gray-900/60 border-gray-700"
              value={versionA}
              onChange={(e) => setVersionA(e.target.value)}
            />
          </div>
          
          <div className="space-y-2">
            <h4 className="font-medium">Version B</h4>
            <Textarea 
              placeholder="Enter alternative version of your post..."
              className="h-36 bg-gray-900/60 border-gray-700"
              value={versionB}
              onChange={(e) => setVersionB(e.target.value)}
            />
          </div>
        </div>
        
        <div className="flex justify-center my-4">
          <Button 
            onClick={handleCompare}
            disabled={isAnalyzing || !versionA.trim() || !versionB.trim()}
            className="px-8"
          >
            {isAnalyzing ? "Analyzing..." : "Compare Versions"}
          </Button>
        </div>
        
        {results && (
          <div className="mt-6 p-4 bg-gray-800/50 rounded-md border border-gray-700 animate-fade-in">
            <h4 className="font-medium mb-4 text-center">Results</h4>
            
            <div className="flex justify-between items-center mb-6">
              <div className={`p-3 rounded ${results.winnerVersion === "A" ? "bg-green-900/30 border border-green-700" : "bg-gray-700/30"} text-center w-32`}>
                <p className="text-sm font-medium">Version A</p>
                <p className="text-lg font-bold">{results.scoreA}%</p>
              </div>
              
              <div className="flex flex-col items-center">
                <Scale size={24} className="text-blue-400 mb-1" />
                <ArrowRight size={24} className={`text-green-400 ${results.winnerVersion === "B" ? "rotate-180" : ""}`} />
              </div>
              
              <div className={`p-3 rounded ${results.winnerVersion === "B" ? "bg-green-900/30 border border-green-700" : "bg-gray-700/30"} text-center w-32`}>
                <p className="text-sm font-medium">Version B</p>
                <p className="text-lg font-bold">{results.scoreB}%</p>
              </div>
            </div>
            
            <div className="p-3 bg-gray-900/40 rounded border border-gray-700">
              <h5 className="text-sm font-medium mb-2">Why Version {results.winnerVersion} is predicted to perform better:</h5>
              <ul className="space-y-1">
                {results.reasons.map((reason, index) => (
                  <li key={index} className="text-sm flex items-start gap-2">
                    <span className="text-green-500 mt-1">•</span>
                    <span>{reason}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
        
        <div className="mt-4 p-3 bg-blue-900/20 border border-blue-700/50 rounded-md">
          <p className="text-sm text-blue-300">
            This feature uses AI to predict which version is likely to perform better based on viral post patterns and engagement signals.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
