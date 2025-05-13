import { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  Sparkles,
  ChevronDown,
  ArrowUp,
  BarChart,
  Info,
  X,
} from "lucide-react";
import { toast } from "@/components/ui/sonner";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogClose,
} from "@/components/ui/dialog";
import ApiKeyInput from "@/components/ApiKeyInput";
import PostAnalysisResults from "@/components/PostAnalysisResults";
import {
  analyzePost,
  rewritePost,
  PostAnalysis,
  OpenAIModel,
} from "@/utils/aiService";
import { ThemeToggle } from "@/components/ThemeToggle";
import { ModelSelector } from "@/components/ModelSelector";
import { CharacterCountdown } from "@/components/CharacterCountdown";
import { EmojiToggle } from "@/components/EmojiToggle";
import MainNav from "@/components/navigation/MainNav";


// Constants for dropdown options
const GOALS = [
  { value: "engagement", label: "Engagement (Likes, Replies)" },
  { value: "reach", label: "Reach (Views)" },
  { value: "clicks", label: "Traffic (Clicks)" },
];

const POST_STYLES = [
  { value: "viral", label: "Viral Thread" },
  { value: "controversial", label: "Controversial" },
  { value: "informative", label: "Informative" },
  { value: "inspiring", label: "Inspiring" },
  { value: "humorous", label: "Humorous" },
];

const INDUSTRIES = [
  { value: "saas", label: "SaaS" },
  { value: "ecommerce", label: "E-commerce" },
  { value: "media", label: "Media" },
  { value: "tech", label: "Technology" },
  { value: "finance", label: "Finance" },
  { value: "health", label: "Healthcare" },
];

const MAX_CHARACTERS = 280;
const LOCAL_STORAGE_KEY = "x_post_roast_api_key";
const SESSION_STORAGE_KEY = "x_post_roast_draft";

/**
 * Main component for the X Post Analysis and Rewrite application.
 *
 * This component handles rendering of various sections such as header, post editor,
 * analysis results, and navigation. It manages state for user inputs like post content,
 * selected model, API key, and industry. The component also provides functionality
 * to rewrite posts using AI models, analyze their viral potential, and display the results.
 *
 * @component
 */
const Index = () => {
  // State management
  const [post, setPost] = useState(() => {
    // Try to load draft from sessionStorage on initial render
    if (typeof window !== "undefined") {
      return sessionStorage.getItem(SESSION_STORAGE_KEY) || "";
    }
    return "";
  });

  const [includesMedia, setIncludesMedia] = useState(false);
  const [limitRewrite, setLimitRewrite] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isRewriting, setIsRewriting] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [goal, setGoal] = useState("engagement");
  const [postStyle, setPostStyle] = useState("viral");
  const [industry, setIndustry] = useState("saas");
  const [analysisResults, setAnalysisResults] = useState<PostAnalysis | null>(
    null
  );
  const [originalPost, setOriginalPost] = useState("");
  const [selectedModel, setSelectedModel] =
    useState<OpenAIModel>("gpt-4o-mini");
  const [includeEmojis, setIncludeEmojis] = useState(true);
  const [showTips, setShowTips] = useState(false);

  const [apiKey, setApiKey] = useState(() => {
    // Try to load API key from localStorage on initial render
    if (typeof window !== "undefined") {
      return localStorage.getItem(LOCAL_STORAGE_KEY) || "";
    }
    return "";
  });

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-focus the textarea on load
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }, []);

  // Save draft to sessionStorage whenever post changes
  useEffect(() => {
    if (typeof window !== "undefined") {
      sessionStorage.setItem(SESSION_STORAGE_KEY, post);
    }
  }, [post]);

  // Handlers
  const handlePostChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setPost(e.target.value);
    },
    []
  );

  const handleRewrite = useCallback(async () => {
    if (!post.trim()) {
      toast.error("Please enter a post to rewrite");
      return;
    }

    if (!apiKey) {
      toast.error("Please set your API key to use the rewrite feature");
      return;
    }

    setIsRewriting(true);

    try {
      const rewrittenPost = await rewritePost(
        post,
        postStyle,
        limitRewrite,
        apiKey,
        selectedModel,
        includeEmojis
      );

      setPost(rewrittenPost);
      toast.success("Post rewritten successfully!");
    } catch (error) {
      console.error("Rewrite error:", error);
      // Error toasts are already handled in the aiService.ts file
    } finally {
      setIsRewriting(false);
    }
  }, [post, postStyle, limitRewrite, apiKey, selectedModel, includeEmojis]);

  const handleRoast = useCallback(async () => {
    if (!post.trim()) {
      toast.error("Please enter a post to analyze");
      return;
    }

    if (!apiKey) {
      toast.error("Please set your API key to use the analysis feature");
      return;
    }

    setIsAnalyzing(true);
    setOriginalPost(post);

    try {
      const results = await analyzePost(
        post,
        includesMedia,
        apiKey,
        selectedModel,
        industry,
        goal
      );
      setAnalysisResults(results);
      setShowResults(true);
    } catch (error) {
      console.error("Analysis error:", error);
      // Error toasts are already handled in the aiService.ts file
    } finally {
      setIsAnalyzing(false);
    }
  }, [post, includesMedia, apiKey, selectedModel, industry, goal]);

  const handleBackToEdit = useCallback(() => {
    setShowResults(false);
  }, []);

  const handleClearPost = useCallback(() => {
    setPost("");
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }, []);

  // UI Components
  const renderHeader = () => (
    <>
      {/* Header with Theme Toggle and Emoji Toggle */}
      <div className="w-full flex justify-between mb-4 items-center">
        <EmojiToggle
          includeEmojis={includeEmojis}
          setIncludeEmojis={setIncludeEmojis}
        />
        <ThemeToggle />
      </div>

      {/* Logo and Title */}
      <h1 className="text-4xl md:text-5xl font-bold text-white mb-2 flex items-center">
        <span className="x-logo mr-4">X</span>
        <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">
          Post Roast
        </span>
      </h1>

      {/* Subtitle */}
      <p className="text-gray-400 text-lg mb-8 text-center">
        Get AI-powered insights to improve your posts' engagement and reach
      </p>
    </>
  );

  /**
   * Renders a post editor component with various input fields and actions.
   */
  const renderPostEditor = () => (
    <>
      {/* Primary Goal */}
      <div className="w-full mb-4">
        <div className="text-gray-400 mb-2 flex items-center">
          Primary Goal
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info size={16} className="ml-2 text-gray-500 cursor-help" />
              </TooltipTrigger>
              <TooltipContent className="bg-gray-800 text-white p-2 max-w-xs">
                Select the main outcome you want to achieve with your post
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <Select value={goal} onValueChange={setGoal}>
          <SelectTrigger className="w-full bg-secondary text-primary px-4 py-2 border-gray-700 hover:bg-gray-700/50 transition-colors">
            <SelectValue placeholder="Select a goal" />
          </SelectTrigger>
          <SelectContent className="bg-secondary border-gray-700">
            {GOALS.map((g) => (
              <SelectItem key={g.value} value={g.value}>
                {g.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Media Toggle */}
      <div className="w-full mb-4 flex items-center justify-end">
        <div className="flex items-center space-x-2">
          <span className="text-gray-400">Post includes image or video</span>
          <Switch checked={includesMedia} onCheckedChange={setIncludesMedia} />
        </div>
      </div>

      {/* Post Textarea */}
      <div className="w-full mb-4 relative group">
        <div className="absolute top-3 right-3 flex gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0 text-gray-400 hover:text-white hover:bg-gray-700/50"
            onClick={handleClearPost}
            disabled={!post}
          >
            <X size={16} />
            <span className="sr-only">Clear</span>
          </Button>

          <Dialog open={showTips} onOpenChange={setShowTips}>
            <DialogTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="h-8 w-8 p-0 text-gray-400 hover:text-white hover:bg-gray-700/50"
              >
                <Info size={16} />
                <span className="sr-only">Tips</span>
              </Button>
            </DialogTrigger>
            <DialogContent className="bg-secondary text-primary">
              <DialogHeader>
                <DialogTitle>Tips for Viral Posts</DialogTitle>
                <DialogDescription>
                  Use these strategies to increase engagement
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-2 text-sm">
                <p>• Start with a hook that creates curiosity</p>
                <p>• Use short paragraphs (1-2 sentences)</p>
                <p>• Include a question to encourage replies</p>
                <p>• Create contrast with emojis or punctuation</p>
                <p>• End with a clear call to action</p>
              </div>
              <DialogClose asChild>
                <Button className="w-full mt-2">Got it</Button>
              </DialogClose>
            </DialogContent>
          </Dialog>
        </div>

        <Textarea
          ref={textareaRef}
          className="bg-secondary border-gray-700 h-40 resize-none p-4 text-white transition-all focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          placeholder="What's on your mind? Paste or type your X post here..."
          value={post}
          onChange={handlePostChange}
        />
        <div className="absolute bottom-3 right-3">
          <CharacterCountdown
            currentCount={post.length}
            maxCount={MAX_CHARACTERS}
          />
        </div>
      </div>

      {/* Model and Key Section */}
      <div className="w-full flex flex-wrap justify-between items-center mb-6 gap-2">
        <div className="flex-grow">
          <ModelSelector
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
          />
        </div>

        <ApiKeyInput apiKey={apiKey} setApiKey={setApiKey} />

        <div className="flex gap-2">
          <Select value={industry} onValueChange={setIndustry}>
            <SelectTrigger className="bg-secondary text-primary px-4 py-2 border-gray-700 w-32 hover:bg-gray-700/50 transition-colors">
              <SelectValue placeholder="Select industry" />
            </SelectTrigger>
            <SelectContent className="bg-secondary border-gray-700">
              {INDUSTRIES.map((ind) => (
                <SelectItem key={ind.value} value={ind.value}>
                  {ind.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="w-full flex justify-between items-center gap-4 mb-4">
        <div className="flex-grow">
          <Select value={postStyle} onValueChange={setPostStyle}>
            <SelectTrigger className="w-full bg-secondary text-primary px-4 py-2 border-gray-700 hover:bg-gray-700/50 transition-colors">
              <SelectValue placeholder="Choose viral style for rewrite..." />
            </SelectTrigger>
            <SelectContent className="bg-secondary border-gray-700">
              {POST_STYLES.map((style) => (
                <SelectItem key={style.value} value={style.value}>
                  {style.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex gap-2">
          <Button
            className="bg-secondary hover:bg-accent text-primary px-6 transition-colors hover:bg-blue-900/30"
            onClick={handleRewrite}
            disabled={isRewriting || post.trim() === "" || !apiKey}
          >
            <Sparkles size={16} className="mr-2" />
            {isRewriting ? "Rewriting..." : "Rewrite"}
          </Button>

          <Button
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 transition-colors"
            onClick={handleRoast}
            disabled={isAnalyzing || post.trim() === "" || !apiKey}
          >
            <BarChart size={16} className="mr-2" />
            {isAnalyzing ? "Analyzing..." : "Roast"}
          </Button>
        </div>
      </div>

      {/* Character Limit Toggle */}
      <div className="w-full flex items-center space-x-2">
        <Switch checked={limitRewrite} onCheckedChange={setLimitRewrite} />
        <span className="text-gray-400">
          Limit rewrite to 280 characters (for non-X Premium)
        </span>
      </div>
    </>
  );

  const renderAnalysisResults = () => (
    <>
      <div className="w-full mb-6 flex justify-between items-center">
        <Button
          variant="outline"
          className="border-gray-700 bg-secondary"
          onClick={handleBackToEdit}
        >
          ← Back to Editor
        </Button>

        <h2 className="text-xl font-semibold text-white">Analysis Results</h2>
      </div>
      <PostAnalysisResults
        post={post}
        analysis={analysisResults!}
        includesMedia={includesMedia}
        originalPost={originalPost}
        industry={industry}
        goal={goal}
      />
    </>
  );

  return (
    <div className="min-h-screen grid-bg flex flex-col items-center pt-20 px-4">
      <MainNav />
      <div className="w-full max-w-3xl flex flex-col items-center">
        {renderHeader()}

        {showResults && analysisResults
          ? renderAnalysisResults()
          : renderPostEditor()}
      </div>
    </div>
  );
};

export default Index;
