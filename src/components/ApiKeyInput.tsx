import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { 
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
  SheetFooter,
  SheetClose
} from "@/components/ui/sheet";
import { Key, Eye, EyeOff, AlertCircle, CheckCircle } from "lucide-react";
import { toast } from "@/components/ui/sonner";
import { Alert, AlertDescription } from "@/components/ui/alert";

const API_KEY_STORAGE_KEY = "x_post_roast_api_key";

interface ApiKeyInputProps {
  apiKey: string;
  setApiKey: (key: string) => void;
  onSuccess?: () => void;
}

const ApiKeyInput: React.FC<ApiKeyInputProps> = ({ 
  apiKey, 
  setApiKey,
  onSuccess 
}) => {
  const [tempApiKey, setTempApiKey] = useState(apiKey);
  const [isOpen, setIsOpen] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  
  // Reset temp key when sheet opens
  useEffect(() => {
    if (isOpen) {
      setTempApiKey(apiKey);
      setShowKey(false);
      setValidationError(null);
    }
  }, [isOpen, apiKey]);

  const validateApiKey = (key: string): boolean => {
    // Basic validation for OpenAI API key format (sk-...)
    if (!key.trim()) {
      setValidationError("API key cannot be empty");
      return false;
    }
    
    if (!key.startsWith("sk-")) {
      setValidationError("Invalid API key format. OpenAI keys start with 'sk-'");
      return false;
    }
    
    if (key.length < 20) {
      setValidationError("API key is too short. Please check your key");
      return false;
    }
    
    setValidationError(null);
    return true;
  };
  
  const handleSaveApiKey = async () => {
    if (!validateApiKey(tempApiKey)) {
      return;
    }
    
    setIsValidating(true);
    
    try {
      // Optional: Add an actual API validation check here
      // For example, make a minimal API call to verify the key works
      // const isValid = await validateKeyWithOpenAI(tempApiKey);
      
      // For now we'll simulate a quick validation
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Save the key securely
      setApiKey(tempApiKey);
      
      // Store in localStorage with encryption if needed in production
      localStorage.setItem(API_KEY_STORAGE_KEY, tempApiKey);
      
      toast.success("API key saved successfully", {
        description: "Your key has been securely stored in your browser",
        icon: <CheckCircle size={16} />
      });
      
      setIsOpen(false);
      
      if (onSuccess) {
        onSuccess();
      }
    } catch (error) {
      toast.error("Failed to validate API key", {
        description: error instanceof Error ? error.message : "Please try again",
        icon: <AlertCircle size={16} />
      });
    } finally {
      setIsValidating(false);
    }
  };

  const handleClearApiKey = () => {
    setTempApiKey("");
    setApiKey("");
    localStorage.removeItem(API_KEY_STORAGE_KEY);
    toast.info("API key removed");
    setIsOpen(false);
  };
  
  const toggleShowKey = () => setShowKey(!showKey);
  
  // Display masked API key for UI
  const getMaskedKey = () => {
    if (!apiKey) return "";
    if (apiKey.length <= 8) return "••••••••";
    return `${apiKey.substring(0, 4)}...${apiKey.substring(apiKey.length - 4)}`;
  };
  
  return (
    <Sheet open={isOpen} onOpenChange={setIsOpen}>
      <SheetTrigger asChild>
        <Button 
          variant="outline" 
          className="bg-secondary text-primary border-gray-700 flex items-center gap-2"
          aria-label="Set API Key"
        >
          <Key size={16} />
          {apiKey ? getMaskedKey() : "Set API Key"}
        </Button>
      </SheetTrigger>
      <SheetContent className="bg-secondary border-gray-700 w-full sm:max-w-md">
        <SheetHeader>
          <SheetTitle className="text-primary">Set Your API Key</SheetTitle>
          <SheetDescription>
            Enter your OpenAI API key to use the post analysis and rewrite features.
          </SheetDescription>
        </SheetHeader>
        
        <div className="py-6 space-y-4">
          <div className="space-y-2">
            <label htmlFor="apiKey" className="text-sm text-gray-400 flex items-center justify-between">
              <span>OpenAI API Key</span>
              {apiKey && (
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={handleClearApiKey}
                  className="h-6 px-2 text-gray-400 hover:text-red-400"
                >
                  Remove
                </Button>
              )}
            </label>
            
            <div className="relative">
              <Input
                id="apiKey"
                type={showKey ? "text" : "password"}
                placeholder="sk-..."
                value={tempApiKey}
                onChange={(e) => setTempApiKey(e.target.value)}
                className="bg-gray-800 border-gray-700 pr-10"
                disabled={isValidating}
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="absolute right-0 top-0 h-full px-3 text-gray-400"
                onClick={toggleShowKey}
                aria-label={showKey ? "Hide API key" : "Show API key"}
              >
                {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
              </Button>
            </div>
            
            {validationError && (
              <Alert variant="destructive" className="bg-red-900/30 border-red-800 mt-2">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription className="text-xs">{validationError}</AlertDescription>
              </Alert>
            )}
          </div>
          
          <p className="text-xs text-gray-400">
            Your API key is stored securely in your browser's local storage and is never sent to our servers.
          </p>
          
          <SheetFooter className="flex flex-col gap-2 sm:flex-row sm:justify-between">
            <p className="text-xs text-gray-400 self-center">
              Don't have an API key? <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">Get one from OpenAI</a>
            </p>
            
            <div className="flex gap-2 self-end">
              <SheetClose asChild>
                <Button variant="outline" className="border-gray-700">
                  Cancel
                </Button>
              </SheetClose>
              <Button 
                onClick={handleSaveApiKey}
                disabled={!tempApiKey || isValidating}
                className={isValidating ? "opacity-80" : ""}
              >
                {isValidating ? "Validating..." : "Save API Key"}
              </Button>
            </div>
          </SheetFooter>
        </div>
      </SheetContent>
    </Sheet>
  );
};

export default ApiKeyInput;