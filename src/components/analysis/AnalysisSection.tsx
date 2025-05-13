import { LucideIcon } from "lucide-react";

interface AnalysisSectionProps {
  title: string;
  items: string[];
  icon: LucideIcon;
  iconColor: string;
}

export function AnalysisSection({ title, items, icon: Icon, iconColor }: AnalysisSectionProps) {
  if (!items || items.length === 0) return null;

  return (
    <div className="mb-6">
      <div className="flex items-start gap-2 mb-2">
        <div className="mt-1 text-white">
          <Icon size={16} className={iconColor} />
        </div>
        <div>
          <h4 className={`font-medium ${iconColor}`}>{title}</h4>
          <ul className="mt-2 space-y-2 text-sm text-gray-300">
            {items.map((item, index) => (
              <li key={index} className="animate-fade-in" style={{ animationDelay: `${index * 100}ms` }}>
                • {item}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
