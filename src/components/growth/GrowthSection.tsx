import { HookLibrary } from "./HookLibrary";
import { ABTesting } from "./ABTesting";

interface GrowthSectionProps {
  post: string;
  includeEmojis: boolean;
}

export function GrowthSection({ post, includeEmojis }: GrowthSectionProps) {
  const growthIcon = includeEmojis ? "📈 " : "";

  return (
    <div className="w-full space-y-8">
      <div className="w-full flex flex-col items-center justify-center p-6 bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-lg border border-purple-800/50">
        <h2 className="text-2xl font-bold mb-2">
          {growthIcon}Growth & Scheduling
        </h2>
        <p className="text-gray-300 text-center mb-6 max-w-lg">
          Tools to improve engagement, test variations, and schedule your posts
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="col-span-2">
          <HookLibrary includeEmojis={includeEmojis} />
        </div>
      </div>

      <ABTesting includeEmojis={includeEmojis} />
    </div>
  );
}
