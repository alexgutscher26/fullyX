import { useState } from "react";
import { BestTimeToPost } from "./BestTimeToPost";
import { ViralityScore } from "./ViralityScore";
import { AIRewriter } from "./AIRewriter";
import { AccountAnalysis } from "./AccountAnalysis";
import { CompetitorComparison } from "./CompetitorComparison";
import { ExportFeature } from "./ExportFeature";
import { Sparkles, Lock, ChevronRight, Crown, Diamond, Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "@/components/ui/sonner";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";

interface PremiumFeaturesSectionProps {
  post: string;
  includeEmojis: boolean;
}

export function PremiumFeaturesSection({ post, includeEmojis }: PremiumFeaturesSectionProps) {
  const premiumIcon = includeEmojis ? "💎 " : "";
  const [activeTab, setActiveTab] = useState<string>("optimizer");
  const [isPremiumModalOpen, setIsPremiumModalOpen] = useState(false);
  
  const handlePremiumFeatureClick = (featureName: string) => {
    toast.info("Premium Feature", {
      description: `'${featureName}' requires a premium subscription.`,
      action: {
        label: "Upgrade",
        onClick: () => handleUpgradeClick()
      },
    });
  };
  
  const handleUpgradeClick = () => {
    setIsPremiumModalOpen(true);
    
    toast.info("Premium Feature", {
      description: "This would redirect to the premium upgrade page in a real app.",
      action: {
        label: "Learn More",
        onClick: () => toast("Redirecting to premium features page...")
      },
    });
    
    // Simulate closing the modal after a delay
    setTimeout(() => setIsPremiumModalOpen(false), 3000);
  };
  
  return (
    <div className="w-full space-y-8 animate-fade-in">
      <div className="w-full relative overflow-hidden flex flex-col items-center justify-center p-8 bg-gradient-to-br from-blue-900/40 to-purple-900/40 rounded-xl border border-purple-800/30 shadow-lg backdrop-blur-sm">
        {/* Background effects */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-0 left-0 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl"></div>
          <div className="absolute bottom-0 right-0 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl"></div>
        </div>
        
        <div className="relative">
          <div className="h-16 w-16 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center mb-6 shadow-lg transform hover:scale-110 transition-all group">
            <Diamond size={32} className="text-white group-hover:animate-pulse" />
          </div>
          
          <h2 className="text-3xl font-bold mb-3 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
            {premiumIcon}Premium Features
          </h2>
          <p className="text-gray-300 text-center mb-6 max-w-lg">
            Unlock advanced analytics and AI-powered tools to optimize your X posts and maximize engagement.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center gap-3">
            <Button 
              onClick={handleUpgradeClick}
              className="px-6 py-6 text-lg flex items-center gap-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 shadow-md border border-purple-500/30 hover:shadow-lg transition-all"
            >
              <Crown size={18} className="mr-1" />
              Upgrade to Premium
              <ChevronRight size={16} />
            </Button>
            
            <Button 
              variant="outline"
              onClick={() => toast.info("Free trial activated!", { description: "Enjoy 7 days of premium features." })}
              className="px-6 py-6 text-lg border-purple-500/40 bg-purple-900/20 hover:bg-purple-900/30 text-purple-300 transition-all"
            >
              <Star size={18} className="mr-1" />
              Start Free Trial
            </Button>
          </div>
          
          {isPremiumModalOpen && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm rounded-xl z-10">
              <div className="bg-gradient-to-br from-blue-900/90 to-purple-900/90 p-6 rounded-lg border border-purple-500/40 shadow-lg max-w-md animate-fade-in-up">
                <h3 className="text-xl font-bold mb-2 text-center">Choose Your Premium Plan</h3>
                <p className="text-gray-300 text-center mb-4">Select the plan that's right for you</p>
                <div className="text-center">
                  <Button 
                    className="w-full mb-2 bg-gradient-to-r from-blue-600 to-purple-600"
                    onClick={() => setIsPremiumModalOpen(false)}
                  >
                    Continue to checkout
                  </Button>
                  <Button 
                    variant="ghost" 
                    className="text-gray-400"
                    onClick={() => setIsPremiumModalOpen(false)}
                  >
                    Cancel
                  </Button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
      
      <Tabs defaultValue="optimizer" className="w-full">
        <TabsList className="w-full grid grid-cols-3 mb-6 bg-secondary/50 p-1 rounded-lg">
          <TabsTrigger 
            value="optimizer" 
            className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-purple-600 data-[state=active]:text-white"
            onClick={() => setActiveTab("optimizer")}
          >
            AI Tools
          </TabsTrigger>
          <TabsTrigger 
            value="analytics" 
            className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-purple-600 data-[state=active]:text-white"
            onClick={() => setActiveTab("analytics")}
          >
            Analytics
          </TabsTrigger>
          <TabsTrigger 
            value="extras" 
            className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-purple-600 data-[state=active]:text-white"
            onClick={() => setActiveTab("extras")}
          >
            Extras
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="optimizer" className="space-y-6 animate-fade-in">
          <div className="relative rounded-lg overflow-hidden">
            <div onClick={() => handlePremiumFeatureClick("AI Rewriter")} className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-10 cursor-pointer hover:bg-black/50 transition-all">
              <Button className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-purple-600">
                <Lock size={16} />
                Unlock Premium Feature
              </Button>
            </div>
            <AIRewriter post={post} includeEmojis={includeEmojis} />
          </div>
        </TabsContent>
        
        <TabsContent value="analytics" className="space-y-6 animate-fade-in">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="relative rounded-lg overflow-hidden">
              <div onClick={() => handlePremiumFeatureClick("Best Time to Post")} className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-10 cursor-pointer hover:bg-black/50 transition-all">
                <Button className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-purple-600">
                  <Lock size={16} />
                  Unlock Premium Feature
                </Button>
              </div>
              <BestTimeToPost includeEmojis={includeEmojis} />
            </div>
            <div className="relative rounded-lg overflow-hidden">
              <div onClick={() => handlePremiumFeatureClick("Virality Score")} className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-10 cursor-pointer hover:bg-black/50 transition-all">
                <Button className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-purple-600">
                  <Lock size={16} />
                  Unlock Premium Feature
                </Button>
              </div>
              <ViralityScore includeEmojis={includeEmojis} />
            </div>
          </div>
          <div className="relative rounded-lg overflow-hidden">
            <div onClick={() => handlePremiumFeatureClick("Account Analysis")} className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-10 cursor-pointer hover:bg-black/50 transition-all">
              <Button className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-purple-600">
                <Lock size={16} />
                Unlock Premium Feature
              </Button>
            </div>
            <AccountAnalysis includeEmojis={includeEmojis} />
          </div>
        </TabsContent>
        
        <TabsContent value="extras" className="space-y-6 animate-fade-in">
          <div className="relative rounded-lg overflow-hidden">
            <div onClick={() => handlePremiumFeatureClick("Competitor Comparison")} className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-10 cursor-pointer hover:bg-black/50 transition-all">
              <Button className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-purple-600">
                <Lock size={16} />
                Unlock Premium Feature
              </Button>
            </div>
            <CompetitorComparison includeEmojis={includeEmojis} />
          </div>
          <div className="relative rounded-lg overflow-hidden">
            <div onClick={() => handlePremiumFeatureClick("Export Features")} className="absolute inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-10 cursor-pointer hover:bg-black/50 transition-all">
              <Button className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-purple-600">
                <Lock size={16} />
                Unlock Premium Feature
              </Button>
            </div>
            <ExportFeature includeEmojis={includeEmojis} />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
