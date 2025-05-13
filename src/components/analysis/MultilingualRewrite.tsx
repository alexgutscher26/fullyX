import { useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Globe, Languages, RefreshCw, Check, Info, Copy, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { analyzePost, rewritePost, type OpenAIModel, type PostAnalysis } from "@/utils/aiService";
import { toast } from "@/components/ui/sonner";
import { ScoreBar } from "./ScoreBar";

// List of available languages for rewriting
const LANGUAGES = [
  { code: "en", name: "English" },
  { code: "es", name: "Spanish" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "it", name: "Italian" },
  { code: "pt", name: "Portuguese" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
  { code: "zh", name: "Chinese" },
  { code: "ar", name: "Arabic" },
  { code: "hi", name: "Hindi" },
  { code: "ru", name: "Russian" }
];

// Writing styles for rewriting
const WRITING_STYLES = [
  { value: "professional", label: "Professional" },
  { value: "casual", label: "Casual" },
  { value: "enthusiastic", label: "Enthusiastic" },
  { value: "humorous", label: "Humorous" },
  { value: "formal", label: "Formal" },
  { value: "persuasive", label: "Persuasive" }
];

// Post variation with analysis
interface PostVariation {
  id: string;
  language: string;
  text: string;
  analysis: PostAnalysis | null;
  isAnalyzing: boolean;
}

interface MultilingualRewriteProps {
  apiKey: string;
  model: OpenAIModel;
  includeEmojis: boolean;
}

/**
 * The `RewriteAndLocalize` component renders a UI for rewriting and localizing an original post into multiple languages.
 * It provides options to generate variations in different languages, view analysis of each variation, and retry analysis if needed.
 *
 * @component
 * @param {Object} props - The props passed to the component.
 * @param {string} props.originalPost - The original text to be rewritten and localized.
 * @param {Array<Object>} props.variations - An array of objects representing the different language variations of the original post.
 * @param {Function} props.onRewriteAndLocalize - A function to generate variations in multiple languages when "Rewrite & Localize" is clicked.
 * @param {boolean} props.isRewriting - A flag indicating whether the rewriting process is currently running.
 * @param {Array<Object>} props.languages - An array of language objects that can be selected for generating variations.
 * @param {Function} props.onLanguageSelect - A function to handle selection of a language for generating variations.
 * @param {boolean} props.isExpanded - A flag indicating whether the analysis section is expanded or collapsed.
 * @param {Function} props.onToggleExpand - A function to toggle the expansion state of the analysis section.
 *
 */
export function MultilingualRewrite({ apiKey, model, includeEmojis }: MultilingualRewriteProps) {
  // Original post state
  const [originalPost, setOriginalPost] = useState<string>("");
  
  // Modal state
  const [isDialogOpen, setIsDialogOpen] = useState<boolean>(false);
  
  // Selected languages for rewriting
  const [selectedLanguages, setSelectedLanguages] = useState<string[]>(["en"]);
  
  // Writing style and length constraint
  const [writingStyle, setWritingStyle] = useState<string>("professional");
  const [limitLength, setLimitLength] = useState<boolean>(true);
  
  // Post variations after rewriting
  const [variations, setVariations] = useState<PostVariation[]>([]);
  
  // Loading state
  const [isRewriting, setIsRewriting] = useState<boolean>(false);
  
  // Expanded view of analysis
  const [isExpanded, setIsExpanded] = useState<boolean>(true);

  // Toggle language selection
  /**
   * Toggles a language by adding or removing it from the selected languages list.
   */
  const toggleLanguage = (langCode: string) => {
    setSelectedLanguages(prev => {
      if (prev.includes(langCode)) {
        // Don't remove if it's the last selected language
        if (prev.length === 1) return prev;
        return prev.filter(code => code !== langCode);
      } else {
        return [...prev, langCode];
      }
    });
  };

  // Handle rewrite action
  /**
   * Handles the rewriting of a post into selected languages and analyzes each translation.
   */
  const handleRewrite = async () => {
    if (!originalPost.trim()) {
      toast.error("Please enter a post to rewrite");
      return;
    }

    if (!apiKey) {
      toast.error("API key is required for rewriting");
      return;
    }

    setIsRewriting(true);
    
    try {
      // Create placeholder variations
      const newVariations: PostVariation[] = selectedLanguages.map(lang => ({
        id: `${lang}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
        language: lang,
        text: "",
        analysis: null,
        isAnalyzing: false
      }));
      
      setVariations(newVariations);
      setIsDialogOpen(false);
      
      // Process each language in parallel
      await Promise.all(
        selectedLanguages.map(async (langCode, index) => {
          try {
            // Get language name for prompt
            const langName = LANGUAGES.find(l => l.code === langCode)?.name || langCode;
            
            // Rewrite post in the target language
            const rewrittenText = await rewritePost(
              originalPost,
              `${writingStyle} ${langName === "English" ? "" : `in ${langName}`}`,
              limitLength,
              apiKey,
              model,
              includeEmojis
            );
            
            // Update the variation with the rewritten text
            setVariations(prev => {
              const updated = [...prev];
              updated[index] = {
                ...updated[index],
                text: rewrittenText,
                isAnalyzing: true
              };
              return updated;
            });
            
            // Analyze the rewritten post
            const analysis = await analyzePost(
              rewrittenText,
              false, // Assuming no media by default
              apiKey,
              model,
              "tech", // Default industry for tech content
              "engagement", // Default goal for social media posts
              model // Pass the model parameter
            );
            
            // Update with analysis results
            setVariations(prev => {
              const updated = [...prev];
              updated[index] = {
                ...updated[index],
                analysis,
                isAnalyzing: false
              };
              return updated;
            });
          } catch (error) {
            console.error(`Error processing ${langCode}:`, error);
            
            // Update variation with error state
            setVariations(prev => {
              const updated = [...prev];
              updated[index] = {
                ...updated[index],
                text: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
                isAnalyzing: false
              };
              return updated;
            });
          }
        })
      );
    } catch (error) {
      console.error("Rewrite operation failed:", error);
      toast.error(`Rewrite failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsRewriting(false);
    }
  };

  // Copy variation to clipboard
  /**
   * Copies text to the clipboard and shows a success or failure message.
   */
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
      .then(() => toast.success("Copied to clipboard"))
      .catch(() => toast.error("Failed to copy to clipboard"));
  };

  return (
    <Card className="bg-secondary border-gray-700">
      <CardHeader className="flex flex-row items-center justify-between pb-2 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <Globe size={18} className="text-blue-400" />
          <h3 className="text-lg font-semibold">Multilingual Rewrite</h3>
        </div>
        <div className="flex items-center gap-2">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="h-8 w-8"
                >
                  {isExpanded ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>{isExpanded ? "Collapse Analysis" : "Expand Analysis"}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </CardHeader>
      <CardContent className="pt-4">
        <div className="space-y-4">
          <div>
            <Label htmlFor="originalPost">Original Post</Label>
            <Textarea
              id="originalPost"
              placeholder="Enter your post here..."
              className="bg-gray-800 border-gray-700 h-28"
              value={originalPost}
              onChange={(e) => setOriginalPost(e.target.value)}
            />
          </div>
          
          <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
            <DialogTrigger asChild>
              <Button 
                className="w-full bg-blue-600 hover:bg-blue-700"
                disabled={isRewriting || !originalPost.trim()}
              >
                <Languages size={16} className="mr-2" />
                Rewrite & Localize
              </Button>
            </DialogTrigger>
            <DialogContent className="bg-gray-800 text-white border-gray-700 max-w-md">
              <DialogHeader>
                <DialogTitle>Rewrite & Localize Post</DialogTitle>
                <DialogDescription className="text-gray-400">
                  Select target languages and customize how your post will be rewritten
                </DialogDescription>
              </DialogHeader>
              
              <div className="space-y-4 py-2">
                <div>
                  <Label className="mb-2 block">Target Languages</Label>
                  <div className="flex flex-wrap gap-2">
                    {LANGUAGES.map(lang => (
                      <Badge
                        key={lang.code}
                        variant={selectedLanguages.includes(lang.code) ? "default" : "outline"}
                        className={`cursor-pointer ${
                          selectedLanguages.includes(lang.code) 
                            ? "bg-blue-600 hover:bg-blue-700" 
                            : "bg-transparent hover:bg-gray-700"
                        }`}
                        onClick={() => toggleLanguage(lang.code)}
                      >
                        {selectedLanguages.includes(lang.code) && (
                          <Check size={12} className="mr-1" />
                        )}
                        {lang.name}
                      </Badge>
                    ))}
                  </div>
                </div>
                
                <div>
                  <Label htmlFor="writingStyle">Writing Style</Label>
                  <Select
                    value={writingStyle}
                    onValueChange={setWritingStyle}
                  >
                    <SelectTrigger id="writingStyle" className="bg-gray-800 border-gray-700">
                      <SelectValue placeholder="Select a writing style" />
                    </SelectTrigger>
                    <SelectContent className="bg-gray-800 border-gray-700">
                      {WRITING_STYLES.map(style => (
                        <SelectItem key={style.value} value={style.value}>
                          {style.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Switch
                    id="limitLength"
                    checked={limitLength}
                    onCheckedChange={setLimitLength}
                  />
                  <Label htmlFor="limitLength">Limit to 280 characters</Label>
                </div>
              </div>
              
              <DialogFooter>
                <Button
                  variant="outline"
                  onClick={() => setIsDialogOpen(false)}
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleRewrite}
                  className="bg-blue-600 hover:bg-blue-700"
                >
                  <RefreshCw size={16} className="mr-2" />
                  Generate Variations
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
          
          {isRewriting && (
            <div className="p-4 text-center animate-pulse">
              <p className="text-gray-400">Generating post variations...</p>
            </div>
          )}
          
          {variations.length > 0 && (
            <div className="space-y-6 mt-4">
              <Tabs defaultValue={variations[0]?.language} className="w-full">
                <TabsList className="grid" style={{gridTemplateColumns: `repeat(${variations.length}, 1fr)`}}>
                  {variations.map(variation => (
                    <TabsTrigger key={variation.id} value={variation.language}>
                      {LANGUAGES.find(l => l.code === variation.language)?.name || variation.language}
                    </TabsTrigger>
                  ))}
                </TabsList>
                
                {variations.map(variation => (
                  <TabsContent key={variation.id} value={variation.language}>
                    <div className="space-y-4">
                      <div className="p-3 bg-gray-800/50 rounded-md border border-gray-700 relative">
                        <p className="whitespace-pre-wrap">{variation.text}</p>
                        
                        <Button
                          size="icon"
                          variant="ghost"
                          className="absolute top-2 right-2 h-8 w-8 opacity-70 hover:opacity-100"
                          onClick={() => copyToClipboard(variation.text)}
                        >
                          <Copy size={16} />
                        </Button>
                      </div>
                      
                      {isExpanded && variation.analysis && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 animate-fade-in">
                          <div className="space-y-3">
                            <ScoreBar 
                              label="Overall Score" 
                              score={variation.analysis.score} 
                            />
                            <ScoreBar 
                              label="Engagement Score" 
                              score={variation.analysis.engagementScore} 
                            />
                            <ScoreBar 
                              label="Readability Score" 
                              score={variation.analysis.readabilityScore} 
                            />
                          </div>
                        </div>
                      )}
                      
                      {variation.isAnalyzing && (
                        <div className="p-4 bg-gray-800/50 rounded-md border border-gray-700 animate-pulse">
                          <div className="flex items-center gap-2">
                            <RefreshCw size={16} className="text-blue-400 animate-spin" />
                            <p className="text-sm text-gray-400">Analyzing post...</p>
                          </div>
                        </div>
                      )}
                      
                      {!variation.isAnalyzing && !variation.analysis && variation.text && (
                        <div className="p-4 bg-gray-800/50 rounded-md border border-gray-700">
                          <div className="flex items-center gap-2">
                            <Info size={14} className="text-blue-400" />
                            <h4 className="text-sm font-medium">Analysis Unavailable</h4>
                          </div>
                          <Button 
                            className="mt-2 text-xs h-8"
                            variant="outline"
                            onClick={async () => {
                              try {
                                setVariations(prev => {
                                  const updated = [...prev];
                                  const index = updated.findIndex(v => v.id === variation.id);
                                  if (index !== -1) {
                                    updated[index] = {
                                      ...updated[index],
                                      isAnalyzing: true
                                    };
                                  }
                                  return updated;
                                });
                                
                                const analysis = await analyzePost(
                                  variation.text,
                                  false,
                                  apiKey,
                                  model,
                                  "tech", // Default industry for tech content
                                  "engagement", // Default goal for social media posts
                                  model // Pass the model parameter
                                );
                                
                                setVariations(prev => {
                                  const updated = [...prev];
                                  const index = updated.findIndex(v => v.id === variation.id);
                                  if (index !== -1) {
                                    updated[index] = {
                                      ...updated[index],
                                      analysis,
                                      isAnalyzing: false
                                    };
                                  }
                                  return updated;
                                });
                              } catch (error) {
                                console.error("Analysis retry failed:", error);
                                toast.error(`Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
                                
                                setVariations(prev => {
                                  const updated = [...prev];
                                  const index = updated.findIndex(v => v.id === variation.id);
                                  if (index !== -1) {
                                    updated[index] = {
                                      ...updated[index],
                                      isAnalyzing: false
                                    };
                                  }
                                  return updated;
                                });
                              }
                            }}
                          >
                            <RefreshCw size={12} className="mr-1" />
                            Retry Analysis
                          </Button>
                        </div>
                      )}
                      
                      {isExpanded && variation.analysis && !variation.isAnalyzing && (
                        <div className="p-3 bg-gray-800/50 rounded-md border border-gray-700">
                          <div className="flex items-center gap-2">
                            <Info size={14} className="text-blue-400" />
                            <h4 className="text-sm font-medium">Analysis Insights</h4>
                          </div>
                          <div className="space-y-2 text-sm">
                            {variation.analysis.optimizationTips.length > 0 && (
                              <div>
                                <h5 className="font-medium text-blue-400 text-xs mb-1">Optimization Tips</h5>
                                <ul className="list-disc list-inside space-y-1 text-gray-300">
                                  {variation.analysis.optimizationTips.slice(0, 3).map((tip, i) => (
                                    <li key={i} className="text-xs">{tip}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                            
                            {variation.analysis.sentiment && (
                              <div>
                                <h5 className="font-medium text-blue-400 text-xs mb-1">Sentiment</h5>
                                <p className="text-xs text-gray-300">
                                  {variation.analysis.sentiment.tone} ({variation.analysis.sentiment.score}%)
                                  {variation.analysis.sentiment.emotions.length > 0 && 
                                    ` - Emotions: ${variation.analysis.sentiment.emotions.join(', ')}`
                                  }
                                </p>
                              </div>
                            )}
                            
                            {variation.analysis.hashtags && variation.analysis.hashtags.recommended.length > 0 && (
                              <div>
                                <h5 className="font-medium text-blue-400 text-xs mb-1">Recommended Hashtags</h5>
                                <div className="flex flex-wrap gap-1">
                                  {variation.analysis.hashtags.recommended.slice(0, 5).map((tag, i) => (
                                    <span key={i} className="px-2 py-0.5 bg-gray-700 rounded-full text-xs">
                                      {tag}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                          <Button 
                            className="mt-2 text-xs h-8"
                            variant="outline"
                            onClick={async () => {
                              try {
                                setVariations(prev => {
                                  const updated = [...prev];
                                  const index = updated.findIndex(v => v.id === variation.id);
                                  if (index !== -1) {
                                    updated[index] = {
                                      ...updated[index],
                                      isAnalyzing: true
                                    };
                                  }
                                  return updated;
                                });
                                
                                const analysis = await analyzePost(
                                  variation.text,
                                  false,
                                  apiKey,
                                  model,
                                  "tech", // Default industry for tech content
                                  "engagement", // Default goal for social media posts
                                  model // Pass the model parameter
                                );
                                
                                setVariations(prev => {
                                  const updated = [...prev];
                                  const index = updated.findIndex(v => v.id === variation.id);
                                  if (index !== -1) {
                                    updated[index] = {
                                      ...updated[index],
                                      analysis,
                                      isAnalyzing: false
                                    };
                                  }
                                  return updated;
                                });
                              } catch (error) {
                                console.error("Analysis retry failed:", error);
                                toast.error(`Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
                                
                                setVariations(prev => {
                                  const updated = [...prev];
                                  const index = updated.findIndex(v => v.id === variation.id);
                                  if (index !== -1) {
                                    updated[index] = {
                                      ...updated[index],
                                      isAnalyzing: false
                                    };
                                  }
                                  return updated;
                                });
                              }
                            }}
                          >
                            <RefreshCw size={12} className="mr-1" />
                            Refresh Analysis
                          </Button>
                        </div>
                      )}
                    </div>
                  </TabsContent>
                ))}
              </Tabs>
            </div>
          )}

          {/* Error state when no variations are available */}
          {!isRewriting && variations.length === 0 && originalPost.trim() && (
            <div className="p-4 bg-gray-800/50 rounded-md border border-gray-700 text-center">
              <p className="text-gray-400">
                No post variations generated yet. Click "Rewrite & Localize" to create variations in multiple languages.
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}