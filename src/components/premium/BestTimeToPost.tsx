import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Clock, BarChart3, Calendar } from "lucide-react";

interface BestTimeToPostProps {
  includeEmojis?: boolean;
}

export function BestTimeToPost({ includeEmojis = true }: BestTimeToPostProps) {
  const timeIcon = includeEmojis ? "⏰ " : "";
  const calendarIcon = includeEmojis ? "📅 " : "";
  const statsIcon = includeEmojis ? "📊 " : "";
  
  // Mock data for best posting times
  const weekdayTimes = [
    { day: "Monday", time: "10:00 AM", engagement: 78 },
    { day: "Tuesday", time: "11:30 AM", engagement: 85 },
    { day: "Wednesday", time: "2:00 PM", engagement: 92 },
    { day: "Thursday", time: "9:00 AM", engagement: 76 },
    { day: "Friday", time: "3:30 PM", engagement: 88 },
    { day: "Saturday", time: "12:00 PM", engagement: 95 },
    { day: "Sunday", time: "7:00 PM", engagement: 82 },
  ];
  
  const bestDay = weekdayTimes.reduce((prev, current) => 
    (prev.engagement > current.engagement) ? prev : current);
    
  return (
    <Card className="bg-secondary border-gray-700">
      <CardHeader className="flex flex-row items-center justify-between pb-2 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <Clock size={18} className="text-blue-400" />
          <h3 className="text-lg font-semibold">{timeIcon}Best Time to Post</h3>
        </div>
        <span className="px-2 py-1 bg-blue-900/40 text-blue-400 text-xs border border-blue-700/50 rounded-full">
          PREMIUM
        </span>
      </CardHeader>
      <CardContent className="pt-4">
        <div className="space-y-4">
          <div className="p-4 bg-gray-800/50 rounded-md border border-gray-700">
            <h4 className="font-medium mb-3 flex items-center gap-2">
              <Calendar size={16} className="text-blue-500" />
              {calendarIcon}Optimal Posting Schedule
            </h4>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <p className="text-sm text-gray-300 mb-1">Best day to post:</p>
                <div className="p-2 bg-blue-900/30 text-blue-100 rounded flex items-center justify-between">
                  <span className="font-medium">{bestDay.day}</span>
                  <span className="text-xs text-blue-300">{bestDay.engagement}% engagement</span>
                </div>
              </div>
              
              <div className="space-y-2">
                <p className="text-sm text-gray-300 mb-1">Best time to post:</p>
                <div className="p-2 bg-blue-900/30 text-blue-100 rounded flex items-center justify-between">
                  <span className="font-medium">{bestDay.time}</span>
                  <span className="text-xs text-blue-300">Based on follower activity</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="p-4 bg-gray-800/50 rounded-md border border-gray-700">
            <h4 className="font-medium mb-3 flex items-center gap-2">
              <BarChart3 size={16} className="text-green-500" />
              {statsIcon}Weekly Engagement Patterns
            </h4>
            
            <div className="space-y-2">
              {weekdayTimes.map((day) => (
                <div key={day.day} className="flex items-center gap-2">
                  <div className="text-gray-400 w-24 text-sm">{day.day}</div>
                  <div className="flex-1 h-5 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-blue-600 to-purple-600 rounded-full"
                      style={{ width: `${day.engagement}%` }}
                    ></div>
                  </div>
                  <div className="text-gray-300 text-sm w-16">{day.time}</div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="p-3 bg-gray-900/30 rounded-md border border-yellow-700/50 text-center">
            <p className="text-sm text-yellow-400">
              Upgrade to Premium to unlock personalized posting times based on your audience's activity patterns
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
