interface ScoreBarProps {
  label: string;
  score: number;
}

export function ScoreBar({ label, score }: ScoreBarProps) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span>{label}</span>
        <span>{score}%</span>
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div 
          className={`h-full rounded-full transition-all duration-500 ease-out ${
            score < 40 ? "bg-red-500" :
            score < 70 ? "bg-yellow-500" :
            "bg-green-500"
          }`}
          style={{ width: `${score}%` }}
        ></div>
      </div>
    </div>
  );
}
