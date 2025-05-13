import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Smile, Frown, Meh, MessageSquare, Info } from "lucide-react";

const EMOTION_COLORS = {
  joy: "bg-yellow-500",
  happiness: "bg-green-500",
  excitement: "bg-purple-500",
  love: "bg-pink-500",
  anger: "bg-red-600",
  sadness: "bg-blue-500",
  fear: "bg-indigo-600",
  surprise: "bg-cyan-500",
  disgust: "bg-orange-600",
  neutral: "bg-gray-500",
  // Add more emotions with corresponding colors
};

// Get a color for any emotion
const getEmotionColor = (emotion) => {
  const normalizedEmotion = emotion.toLowerCase();
  return EMOTION_COLORS[normalizedEmotion] || "bg-gray-600";
};

const SentimentScore = ({ score }) => {
  const [displayScore, setDisplayScore] = useState(0);
  
  useEffect(() => {
    // Animate the score from 0 to actual value
    const timeout = setTimeout(() => {
      setDisplayScore(score);
    }, 300);
    
    return () => clearTimeout(timeout);
  }, [score]);
  
  // Calculate dashboard offset for the circle animation
  const circumference = 2 * Math.PI * 50;
  const dashOffset = (1 - displayScore / 100) * circumference;
  
  // Determine color based on score
  const scoreColor = 
    score < 30 ? "text-red-500" :
    score < 45 ? "text-orange-500" :
    score < 60 ? "text-yellow-500" :
    score < 75 ? "text-lime-500" :
    "text-green-500";
    
  return (
    <div className="flex items-center justify-center">
      <div className="relative w-32 h-32">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <span className={`text-2xl font-bold ${scoreColor}`}>{Math.round(displayScore)}%</span>
          </div>
        </div>
        <svg className="w-32 h-32 transform -rotate-90">
          <circle
            cx="64"
            cy="64"
            r="50"
            fill="transparent"
            stroke="currentColor"
            strokeWidth="12"
            className="text-gray-700"
          />
          <circle
            cx="64"
            cy="64"
            r="50"
            fill="transparent"
            stroke="currentColor"
            strokeWidth="12"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            strokeLinecap="round"
            className={`transition-all duration-1000 ease-out ${scoreColor}`}
          />
        </svg>
      </div>
    </div>
  );
};

const EmotionTag = ({ emotion, index }) => {
  const [show, setShow] = useState(false);
  
  useEffect(() => {
    const timeout = setTimeout(() => {
      setShow(true);
    }, 150 * index);
    
    return () => clearTimeout(timeout);
  }, [index]);
  
  return (
    <span
      className={`px-3 py-1 ${getEmotionColor(emotion)} rounded-full text-sm font-medium transition-all duration-300 ${
        show ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2"
      }`}
    >
      {emotion}
    </span>
  );
};

export function SentimentAnalysis({ sentiment, includeEmojis = true }) {
  const [showTooltip, setShowTooltip] = useState(false);
  
  if (!sentiment) {
    return (
      <Card className="bg-secondary border-gray-700 shadow-lg">
        <CardHeader className="flex flex-row items-center justify-between pb-2 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <MessageSquare size={18} className="text-blue-400" />
            <h3 className="text-lg font-semibold">Sentiment Analysis</h3>
          </div>
        </CardHeader>
        <CardContent className="pt-6 pb-4">
          <div className="text-center text-gray-400 py-8">
            <MessageSquare size={36} className="mx-auto mb-3 opacity-50" />
            <p>No sentiment data available</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { tone, score, emotions = [], isPositive, isNegative, isNeutral } = sentiment;

  // Icon selection based on sentiment
  const SentimentIcon = isPositive ? Smile : isNegative ? Frown : Meh;
  const iconColor = isPositive ? "text-green-500" : isNegative ? "text-red-500" : "text-yellow-500";
  
  // Emoji selection based on toggle
  const sentimentEmoji = includeEmojis ? (
    isPositive ? "😊 " : isNegative ? "😞 " : "😐 "
  ) : "";
  
  // Get appropriate message based on sentiment
  const analysisMessage = isPositive 
    ? "This content has a positive tone which typically generates better engagement and connection with your audience."
    : isNegative
    ? "This content has a negative tone which might limit engagement. Consider adjusting tone for broader appeal."
    : "This content has a neutral tone. Consider adding more emotional language for better audience engagement.";

  return (
    <Card className="bg-secondary border-gray-700 shadow-lg overflow-hidden">
      <CardHeader className="flex flex-row items-center justify-between pb-2 border-b border-gray-700 bg-gray-800/50">
        <div className="flex items-center gap-2">
          <SentimentIcon size={20} className={iconColor} />
          <h3 className="text-lg font-semibold">Sentiment Analysis</h3>
        </div>
        <div className="relative">
          <button
            onClick={() => setShowTooltip(!showTooltip)}
            onBlur={() => setShowTooltip(false)}
            className="text-gray-400 hover:text-gray-300 focus:outline-none"
            aria-label="Show information about sentiment analysis"
          >
            <Info size={16} />
          </button>
          {showTooltip && (
            <div className="absolute right-0 mt-2 w-64 p-3 bg-gray-900 rounded-md shadow-lg z-10 text-xs border border-gray-700">
              Sentiment analysis evaluates the emotional tone of your content.
              Scores range from 0-100, with higher scores indicating more positive sentiment.
            </div>
          )}
        </div>
      </CardHeader>
      
      <CardContent className="pt-6 pb-4">
        <div className="space-y-6">
          <SentimentScore score={score} />
          
          <div className="text-center">
            <h4 className="font-semibold text-xl capitalize">
              {sentimentEmoji}{tone}
            </h4>
          </div>
          
          {emotions.length > 0 && (
            <div className="p-4 bg-gray-800/50 rounded-md border border-gray-700">
              <h4 className="font-medium mb-3">
                {includeEmojis ? "🧠 " : ""}Detected Emotions
              </h4>
              <div className="flex flex-wrap gap-2">
                {emotions.map((emotion, index) => (
                  <EmotionTag key={emotion} emotion={emotion} index={index} />
                ))}
              </div>
            </div>
          )}
          
          <div className="p-4 bg-gray-800/50 rounded-md border border-gray-700">
            <div className="flex gap-3">
              <SentimentIcon size={20} className={`${iconColor} mt-1`} />
              <p className="text-sm">
                {includeEmojis ? "💭 " : ""}
                {analysisMessage}
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}