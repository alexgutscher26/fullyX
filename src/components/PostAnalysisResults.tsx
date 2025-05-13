import React, { useState, memo } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  CheckCircle,
  AlertTriangle,
  Lightbulb,
  Clock,
  Users,
  KeyRound,
  BarChart,
  TrendingUp,
  Hash,
  Star,
  Share,
} from "lucide-react";
import type { PostAnalysis } from "@/utils/aiService";
import { ScoreBar } from "./analysis/ScoreBar";
import { AnalysisSection } from "./analysis/AnalysisSection";
import { KeyMetric } from "./analysis/KeyMetric";
import { SentimentAnalysis } from "./analysis/SentimentAnalysis";
import { HashtagAnalyzer } from "./analysis/HashtagAnalyzer";
import { EmojiToggle } from "./EmojiToggle";
import { PremiumFeaturesSection } from "./premium/PremiumFeaturesSection";
import { GrowthSection } from "./growth/GrowthSection";
import { CommunitySection } from "./community/CommunitySection";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface AnalysisResultsProps {
  post: string;
  analysis: PostAnalysis;
  includesMedia: boolean;
  originalPost: string;
  industry: string;
  goal: string;
}

// Memoized components for better performance
const MemoizedScoreCircle = memo(({ score }: { score: number }) => (
  <div className="relative w-24 h-24 mx-auto">
    <div className="absolute inset-0 flex items-center justify-center">
      <span className="text-2xl font-bold">{score}%</span>
    </div>
    <svg className="w-24 h-24 transform -rotate-90" aria-hidden="true">
      <circle
        cx="48"
        cy="48"
        r="36"
        fill="transparent"
        stroke="currentColor"
        strokeWidth="8"
        strokeDasharray="226.2"
        strokeDashoffset={(1 - score / 100) * 226.2}
        className={`transition-all duration-1000 ease-out ${
          score < 40
            ? "text-red-500"
            : score < 70
            ? "text-yellow-500"
            : "text-green-500"
        }`}
      />
    </svg>
  </div>
));

const MemoizedOriginalPost = memo(({ post }: { post: string }) => (
  <div className="border border-gray-700 rounded-md p-4 mb-4 bg-gray-800/50">
    <p className="text-sm text-gray-400 mb-2">
      <strong>Original Post:</strong>
    </p>
    <p className="text-sm">{post}</p>
  </div>
));

const MemoizedKeyFactors = memo(({ factors }: { factors: string[] }) => (
  <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm text-gray-300">
    {factors.map((factor, index) => (
      <div
        key={index}
        className="animate-fade-in"
        style={{ animationDelay: `${index * 50}ms` }}
      >
        • {factor}
      </div>
    ))}
  </div>
));

const PostAnalysisResults: React.FC<AnalysisResultsProps> = ({
  post,
  analysis,
  includesMedia,
  originalPost,
  industry,
  goal,
}) => {
  const [includeEmojis, setIncludeEmojis] = useState(true);
  const [activeTab, setActiveTab] = useState("algorithm");

  // Extract values from analysis with fallbacks to prevent undefined issues
  const {
    score = 0,
    engagementScore = 0,
    readabilityScore = 0,
    viralityScore = 0,
    algorithmFriendly = [],
    optimizationOpportunities = [],
    optimizationTips = [],
    keyFactors = [],
    bestPostingTime = { daily: "", peakDays: "" },
    targetAudience = "",
    keywords = "",
    sentiment = {
      tone: "neutral",
      score: 50,
      emotions: [],
      isPositive: false,
      isNegative: false,
      isNeutral: true,
    },
    hashtags = {
      trending: [],
      recommended: [],
      reachPotential: "Low",
      relevanceScore: 0,
    },
  } = analysis || {};

  // Handle tab change
  const handleTabChange = (value: string) => {
    setActiveTab(value);
  };

  return (
    <div className="w-full animate-fade-in">
      <div className="w-full flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">Post Analysis Results</h2>
        <EmojiToggle
          includeEmojis={includeEmojis}
          setIncludeEmojis={setIncludeEmojis}
        />
      </div>

      <Tabs
        value={activeTab}
        onValueChange={handleTabChange}
        className="w-full"
      >
        <TabsList
          className="w-full mb-4 bg-secondary border-gray-700 overflow-x-auto flex-nowrap"
          role="tablist"
        >
          <TabsTrigger
            value="algorithm"
            className="flex items-center gap-1"
            role="tab"
          >
            <BarChart size={16} aria-hidden="true" /> <span>Algorithm</span>
          </TabsTrigger>
          <TabsTrigger
            value="sentiment"
            className="flex items-center gap-1"
            role="tab"
          >
            <Users size={16} aria-hidden="true" /> <span>Sentiment</span>
          </TabsTrigger>
          <TabsTrigger
            value="hashtags"
            className="flex items-center gap-1"
            role="tab"
          >
            <Hash size={16} aria-hidden="true" /> <span>Hashtags</span>
          </TabsTrigger>
          <TabsTrigger
            value="growth"
            className="flex items-center gap-1"
            role="tab"
          >
            <TrendingUp size={16} aria-hidden="true" /> <span>Growth</span>
          </TabsTrigger>
          <TabsTrigger
            value="community"
            className="flex items-center gap-1"
            role="tab"
          >
            <Share size={16} aria-hidden="true" /> <span>Community</span>
          </TabsTrigger>
          <TabsTrigger
            value="premium"
            className="flex items-center gap-1"
            role="tab"
          >
            <Star size={16} aria-hidden="true" /> <span>Premium</span>
          </TabsTrigger>
        </TabsList>

        {/* Algorithm Analysis Tab */}
        <TabsContent value="algorithm" className="mt-0" role="tabpanel">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Main Score Card */}
            <div className="col-span-1">
              <Card className="bg-secondary border-gray-700 h-full">
                <CardHeader className="pb-2">
                  <MemoizedScoreCircle score={score} />
                  <p className="text-center text-muted-foreground mt-2">
                    Overall Score
                  </p>
                </CardHeader>
                <CardContent className="pt-0 space-y-4">
                  <MemoizedOriginalPost post={originalPost} />

                  {/* Score Bars */}
                  <div className="space-y-3">
                    <ScoreBar label="Engagement" score={engagementScore} />
                    <ScoreBar label="Readability" score={readabilityScore} />
                    <ScoreBar label="Virality" score={viralityScore} />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Algorithm Analysis Column */}
            <div className="col-span-1 md:col-span-2">
              <Card className="bg-secondary border-gray-700 h-full">
                <CardHeader className="border-b border-gray-700 pb-2">
                  <h3 className="text-lg font-semibold">Algorithm Analysis</h3>
                </CardHeader>
                <CardContent className="pt-4 space-y-4">
                  <AnalysisSection
                    title="Algorithm Friendly Features"
                    items={algorithmFriendly}
                    icon={CheckCircle}
                    iconColor="text-green-500"
                  />

                  <AnalysisSection
                    title="Algorithm Optimization Opportunities"
                    items={optimizationOpportunities}
                    icon={AlertTriangle}
                    iconColor="text-yellow-500"
                  />

                  <AnalysisSection
                    title="Algorithm Optimization Tips"
                    items={optimizationTips}
                    icon={Lightbulb}
                    iconColor="text-blue-500"
                  />

                  <div className="p-4 bg-gray-800/80 rounded-md border border-gray-700">
                    <h4 className="font-medium mb-2">Key Algorithm Factors</h4>
                    <MemoizedKeyFactors factors={keyFactors} />
                  </div>

                  {/* Industry & Goal Analysis */}
                  <div className="p-4 bg-gray-800/80 rounded-md border border-gray-700">
                    <h4 className="font-medium mb-2">
                      Industry & Goal Analysis
                    </h4>
                    <div className="space-y-3">
                      <div className="p-3 bg-gray-900/50 rounded-md border border-gray-600">
                        <h5 className="text-sm font-medium mb-2 text-blue-400">
                          Industry: {industry || "Not specified"}
                        </h5>
                        <p className="text-sm text-gray-300">
                          {industry
                            ? `Optimized for ${industry} industry best practices and audience expectations.`
                            : "No industry specified. Adding industry context can improve recommendations."}
                        </p>
                      </div>
                      <div className="p-3 bg-gray-900/50 rounded-md border border-gray-600">
                        <h5 className="text-sm font-medium mb-2 text-green-400">
                          Goal: {goal || "Not specified"}
                        </h5>
                        <p className="text-sm text-gray-300">
                          {goal
                            ? `Content aligned with goal: ${goal}`
                            : "No specific goal set. Setting clear goals helps optimize content."}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <KeyMetric
                      title="Best Posting Time"
                      value={`Daily: ${
                        bestPostingTime?.daily || "Not specified"
                      }\nPeak days: ${
                        bestPostingTime?.peakDays || "Not specified"
                      }`}
                      icon={Clock}
                      iconColor="text-green-500"
                    />

                    <KeyMetric
                      title="Target Audience"
                      value={targetAudience || "Not specified"}
                      icon={Users}
                      iconColor="text-orange-500"
                    />

                    <KeyMetric
                      title="Optimal Keywords"
                      value={keywords || "Not specified"}
                      icon={KeyRound}
                      iconColor="text-yellow-500"
                    />
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        {/* Sentiment Analysis Tab */}
        <TabsContent value="sentiment" className="mt-0" role="tabpanel">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="col-span-1">
              <SentimentAnalysis
                sentiment={sentiment}
                includeEmojis={includeEmojis}
              />
            </div>
            <div className="col-span-1">
              <Card className="bg-secondary border-gray-700 h-full">
                <CardHeader className="border-b border-gray-700 pb-2">
                  <h3 className="text-lg font-semibold">Post Content & Tips</h3>
                </CardHeader>
                <CardContent className="pt-4 space-y-4">
                  <MemoizedOriginalPost post={originalPost} />

                  <div className="p-3 bg-gray-800/50 rounded-md border border-gray-700">
                    <h4 className="font-medium mb-2">
                      Sentiment Analysis Tips
                    </h4>
                    <ul className="space-y-2 text-sm">
                      <li className="flex items-start gap-2">
                        <span
                          className="text-green-500 mt-1"
                          aria-hidden="true"
                        >
                          •
                        </span>
                        <span>
                          Positive posts tend to get more engagement but less
                          controversy
                        </span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span
                          className="text-yellow-500 mt-1"
                          aria-hidden="true"
                        >
                          •
                        </span>
                        <span>
                          Neutral posts are good for informational content but
                          may lack emotional connection
                        </span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-red-500 mt-1" aria-hidden="true">
                          •
                        </span>
                        <span>
                          Negative posts can drive engagement through
                          controversy but may harm brand perception
                        </span>
                      </li>
                    </ul>
                  </div>

                  <div className="p-3 bg-gray-800/50 rounded-md border border-gray-700">
                    <h4 className="font-medium mb-2">
                      Suggested Tone Adjustments
                    </h4>
                    <p className="text-sm">
                      {sentiment?.isPositive &&
                        "Your positive tone is good. Consider adding specific emotional words to enhance connection."}
                      {sentiment?.isNegative &&
                        "Consider softening negative statements with solutions or alternatives."}
                      {sentiment?.isNeutral &&
                        "Add emotional language or personal perspectives to increase engagement."}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        {/* Hashtag Analysis Tab */}
        <TabsContent value="hashtags" className="mt-0" role="tabpanel">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="col-span-1">
              <HashtagAnalyzer
                hashtags={hashtags}
                includeEmojis={includeEmojis}
              />
            </div>
            <div className="col-span-1">
              <Card className="bg-secondary border-gray-700 h-full">
                <CardHeader className="border-b border-gray-700 pb-2">
                  <h3 className="text-lg font-semibold">
                    Hashtag Best Practices
                  </h3>
                </CardHeader>
                <CardContent className="pt-4 space-y-4">
                  <div className="p-4 bg-gray-800/50 rounded-md border border-gray-700">
                    <h4 className="font-medium mb-3">Why Hashtags Matter</h4>
                    <ul className="space-y-2 text-sm">
                      <li className="flex items-start gap-2">
                        <span className="text-blue-500 mt-1" aria-hidden="true">
                          •
                        </span>
                        <span>
                          Increases content discoverability by up to 55%
                        </span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-blue-500 mt-1" aria-hidden="true">
                          •
                        </span>
                        <span>
                          Helps your content get included in relevant searches
                        </span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-blue-500 mt-1" aria-hidden="true">
                          •
                        </span>
                        <span>
                          Shows X's algorithm what your content is about
                        </span>
                      </li>
                    </ul>
                  </div>

                  <div className="p-4 bg-gray-800/50 rounded-md border border-gray-700">
                    <h4 className="font-medium mb-3">Hashtag Strategy</h4>
                    <ul className="space-y-2 text-sm">
                      <li className="flex items-start gap-2">
                        <span
                          className="text-green-500 mt-1"
                          aria-hidden="true"
                        >
                          •
                        </span>
                        <span>
                          <strong>Use 1-2 trending hashtags</strong> to tap into
                          current conversations
                        </span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span
                          className="text-green-500 mt-1"
                          aria-hidden="true"
                        >
                          •
                        </span>
                        <span>
                          <strong>Add 1-3 niche hashtags</strong> specific to
                          your content area
                        </span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span
                          className="text-green-500 mt-1"
                          aria-hidden="true"
                        >
                          •
                        </span>
                        <span>
                          <strong>Consider branded hashtags</strong> to build
                          community and recognition
                        </span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-red-500 mt-1" aria-hidden="true">
                          •
                        </span>
                        <span>
                          <strong>Avoid overloading</strong> with too many
                          hashtags (3-5 is optimal for X)
                        </span>
                      </li>
                    </ul>
                  </div>

                  <div className="p-4 bg-gray-800/50 rounded-md border border-gray-700">
                    <h4 className="font-medium mb-3">
                      Current X Hashtag Trends
                    </h4>
                    <p className="text-sm text-gray-400 mb-2">
                      Popular categories:
                    </p>
                    <div className="flex flex-wrap gap-2">
                      <span className="px-2 py-1 bg-blue-900/30 text-blue-400 border border-blue-700/50 rounded-full text-xs">
                        #Tech
                      </span>
                      <span className="px-2 py-1 bg-green-900/30 text-green-400 border border-green-700/50 rounded-full text-xs">
                        #Marketing
                      </span>
                      <span className="px-2 py-1 bg-purple-900/30 text-purple-400 border border-purple-700/50 rounded-full text-xs">
                        #AI
                      </span>
                      <span className="px-2 py-1 bg-red-900/30 text-red-400 border border-red-700/50 rounded-full text-xs">
                        #Business
                      </span>
                      <span className="px-2 py-1 bg-yellow-900/30 text-yellow-400 border border-yellow-700/50 rounded-full text-xs">
                        #Productivity
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        {/* Growth & Scheduling Tab */}
        <TabsContent value="growth" className="mt-0" role="tabpanel">
          <GrowthSection post={post} includeEmojis={includeEmojis} />
        </TabsContent>

        {/* Community & Collaboration Tab */}
        <TabsContent value="community" className="mt-0" role="tabpanel">
          <CommunitySection includeEmojis={includeEmojis} />
        </TabsContent>

        {/* Premium Features Tab */}
        <TabsContent value="premium" className="mt-0" role="tabpanel">
          <PremiumFeaturesSection post={post} includeEmojis={includeEmojis} />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default PostAnalysisResults;
