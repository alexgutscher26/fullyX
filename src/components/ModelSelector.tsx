import { 
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { Sparkles } from "lucide-react";
import { useState, useEffect } from "react";

// Define available models with their display names and capabilities
export type OpenAIModel = "gpt-4o-mini" | "gpt-4o" | "gpt-4.5-preview";

interface ModelOption {
  id: OpenAIModel;
  name: string;
  description: string;
  isNew?: boolean;
}

const MODEL_OPTIONS: ModelOption[] = [
  { 
    id: "gpt-4o-mini", 
    name: "GPT-4o Mini", 
    description: "Fast and cost-effective"
  },
  { 
    id: "gpt-4o", 
    name: "GPT-4o", 
    description: "Balanced performance" 
  },
  { 
    id: "gpt-4.5-preview", 
    name: "GPT-4.5 Preview", 
    description: "Most advanced capabilities",
    isNew: true
  }
];

interface ModelSelectorProps {
  selectedModel: OpenAIModel;
  onModelChange: (model: OpenAIModel) => void;
  disabled?: boolean;
  className?: string;
}

export function ModelSelector({ 
  selectedModel, 
  onModelChange, 
  disabled = false,
  className 
}: ModelSelectorProps) {
  const [mounted, setMounted] = useState(false);
  
  // Handle SSR/hydration
  useEffect(() => {
    setMounted(true);
  }, []);

  // Find the currently selected model option
  const selectedOption = MODEL_OPTIONS.find(option => option.id === selectedModel) || MODEL_OPTIONS[0];

  if (!mounted) {
    return (
      <div className={cn(
        "h-10 w-40 bg-secondary rounded-md animate-pulse",
        className
      )} />
    );
  }

  return (
    <Select 
      defaultValue={selectedModel}
      value={selectedModel}
      onValueChange={(value) => onModelChange(value as OpenAIModel)}
      disabled={disabled}
    >
      <SelectTrigger 
        className={cn(
          "bg-secondary text-primary px-4 py-2 border-gray-700 transition-all",
          "hover:bg-secondary/90 focus:ring-2 focus:ring-primary/20",
          disabled && "opacity-50 cursor-not-allowed",
          className
        )}
      >
        <SelectValue 
          placeholder="Select a model"
          className="flex items-center gap-2"
        >
          <span className="flex items-center gap-2">
            {selectedOption.name}
            {selectedOption.isNew && (
              <Sparkles className="h-4 w-4 text-yellow-400" />
            )}
          </span>
        </SelectValue>
      </SelectTrigger>
      
      <SelectContent className="bg-secondary border-gray-700">
        <SelectGroup>
          <SelectLabel className="text-xs text-gray-400 pb-1">Models</SelectLabel>
          {MODEL_OPTIONS.map((option) => (
            <SelectItem 
              key={option.id} 
              value={option.id}
              className="flex flex-col items-start py-2"
            >
              <div className="flex items-center gap-2">
                {option.name}
                {option.isNew && (
                  <span className="bg-yellow-400/20 text-yellow-400 text-xs px-1.5 py-0.5 rounded-full flex items-center gap-1">
                    <Sparkles className="h-3 w-3" />
                    New
                  </span>
                )}
              </div>
              <span className="text-xs text-gray-400">{option.description}</span>
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
}