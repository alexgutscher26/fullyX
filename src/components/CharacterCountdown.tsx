import React, { useState, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import { AlertCircle } from "lucide-react";

interface CharacterCountdownProps {
  currentCount: number;
  maxCount: number;
  warningThreshold?: number; // percentage threshold to start showing warning (default 80%)
  className?: string;
  hideWhenUnderWarning?: boolean;
  ariaLabel?: string;
}

export function CharacterCountdown({
  currentCount,
  maxCount,
  warningThreshold = 80,
  className,
  hideWhenUnderWarning = false,
  ariaLabel,
}: CharacterCountdownProps) {
  const [animateCount, setAnimateCount] = useState(currentCount);
  const prevCountRef = useRef(currentCount);
  
  const percentage = Math.min((currentCount / maxCount) * 100, 100);
  const isWarning = percentage >= warningThreshold && percentage < 100;
  const isOverLimit = currentCount > maxCount;
  const remaining = maxCount - currentCount;
  
  // Determine status for styling and aria label
  const status = isOverLimit ? "error" : isWarning ? "warning" : "normal";
  
  // Smoothly animate count changes
  useEffect(() => {
    if (currentCount !== prevCountRef.current) {
      setAnimateCount(currentCount);
      prevCountRef.current = currentCount;
    }
  }, [currentCount]);
  
  // Hide when under warning threshold if specified
  if (hideWhenUnderWarning && percentage < warningThreshold) {
    return null;
  }

  const defaultAriaLabel = isOverLimit
    ? `Character limit exceeded by ${Math.abs(remaining)}`
    : `${remaining} characters remaining out of ${maxCount}`;

  return (
    <div 
      className={cn(
        "flex items-center gap-2 text-sm font-medium transition-all duration-300",
        status === "error" ? "text-red-500" : 
        status === "warning" ? "text-amber-500" : 
        "text-gray-400",
        className
      )}
      role="status"
      aria-label={ariaLabel || defaultAriaLabel}
      aria-live={isWarning || isOverLimit ? "polite" : "off"}
    >
      {isOverLimit && (
        <AlertCircle className="w-5 h-5" />
      )}
      
      <div className="flex items-center gap-1">
        <span 
          className={cn(
            "transition-all duration-300 text-base",
            (isWarning || isOverLimit) && "font-bold",
            isOverLimit && "animate-pulse"
          )}
        >
          {animateCount}
        </span>
        <span>/</span>
        <span>{maxCount}</span>
      </div>
      
      {(isWarning || isOverLimit) && (
        <span className="text-xs">
          {isOverLimit 
            ? `${Math.abs(remaining)} over limit`
            : `${remaining} remaining`}
        </span>
      )}
    </div>
  );
}