import { Toggle } from "@/components/ui/toggle";
import { Smile, Frown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface EmojiToggleProps {
  includeEmojis: boolean;
  setIncludeEmojis: (include: boolean) => void;
  disabled?: boolean;
  className?: string;
  tooltipText?: string;
}

export function EmojiToggle({ 
  includeEmojis, 
  setIncludeEmojis, 
  disabled = false,
  className,
  tooltipText = "Toggle emoji usage in generated responses"
}: EmojiToggleProps) {
  return (
    <TooltipProvider>
      <div className={cn(
        "flex items-center gap-3 select-none",
        disabled && "opacity-60",
        className
      )}>
        <label 
          htmlFor="emoji-toggle" 
          className={cn(
            "text-sm font-medium cursor-pointer transition-colors",
            includeEmojis ? "text-primary" : "text-gray-400"
          )}
          onClick={() => !disabled && setIncludeEmojis(!includeEmojis)}
        >
          Include emojis
        </label>
        
        <Tooltip>
          <TooltipTrigger asChild>
            <Toggle
              id="emoji-toggle"
              pressed={includeEmojis}
              onPressedChange={setIncludeEmojis}
              aria-label="Toggle emoji inclusion"
              disabled={disabled}
              className={cn(
                "relative h-8 w-12 rounded-full transition-all duration-200",
                "data-[state=on]:bg-blue-600 data-[state=on]:border-blue-700",
                "data-[state=off]:bg-gray-200 data-[state=off]:border-gray-300 dark:data-[state=off]:bg-gray-700 dark:data-[state=off]:border-gray-600",
                "border-2 hover:bg-opacity-90",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2",
                "disabled:cursor-not-allowed"
              )}
            >
              <div className={cn(
                "absolute top-1/2 -translate-y-1/2 flex items-center justify-center w-6 h-6 rounded-full bg-white shadow-sm transition-all duration-200",
                includeEmojis ? "left-5" : "left-1",
                "dark:bg-gray-100"
              )}>
                {includeEmojis ? (
                  <Smile size={16} className="text-blue-600" />
                ) : (
                  <Frown size={16} className="text-gray-400" />
                )}
              </div>
            </Toggle>
          </TooltipTrigger>
          <TooltipContent side="top" className="bg-gray-800 text-white text-xs p-2">
            {tooltipText}
          </TooltipContent>
        </Tooltip>
      </div>
    </TooltipProvider>
  );
}