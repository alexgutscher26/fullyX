import { LucideIcon } from "lucide-react";
interface KeyMetricProps {
  title: string;
  value: string;
  icon: LucideIcon;
  iconColor: string;
}

export function KeyMetric({ title, value, icon: Icon, iconColor }: KeyMetricProps) {
  return (
    <div className="flex items-start gap-2">
      <div className="mt-1">
        <Icon size={16} className={iconColor} />
      </div>
      <div>
        <h4 className="font-medium">{title}</h4>
        <p className="text-xs text-gray-400 break-words">{value || "Not specified"}</p>
      </div>
    </div>
  );
}
