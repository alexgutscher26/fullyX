import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Users, Lock, Globe, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "@/components/ui/sonner";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { useState } from "react";

interface TeamMember {
  id: string;
  initials: string;
  name: string;
  role: string;
  status: "active" | "pending";
  email: string;
}

interface CommunitySectionProps {
  includeEmojis: boolean;
}

export function CommunitySection({ includeEmojis }: CommunitySectionProps) {
  const teamIcon = includeEmojis ? "👥 " : "";
  const clockIcon = includeEmojis ? "⏰ " : "";
  const [email, setEmail] = useState("");
  const [teamMembers, setTeamMembers] = useState<TeamMember[]>([]);
  const [isInviting, setIsInviting] = useState(false);

  const validateEmail = (email: string) => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  };

  const getInitials = (name: string) => {
    return name
      .split(" ")
      .map(part => part[0])
      .join("")
      .toUpperCase();
  };

  const handleInviteTeam = async () => {
    if (!validateEmail(email)) {
      toast.error("Invalid email address");
      return;
    }

    setIsInviting(true);

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));

      const [username] = email.split("@");
      const name = username
        .split(/[._-]/)
        .map(part => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");

      const newMember: TeamMember = {
        id: Date.now().toString(),
        initials: getInitials(name),
        name,
        role: "Viewer",
        status: "pending",
        email
      };

      setTeamMembers(prev => [...prev, newMember]);
      setEmail("");
      toast.success("Invitation sent successfully");
    } catch (error) {
      toast.error("Failed to send invitation");
    } finally {
      setIsInviting(false);
    }
  };

  return (
    <div className="w-full space-y-8 animate-fade-in">
      <div className="w-full flex flex-col items-center justify-center p-8 bg-gradient-to-r from-blue-900/30 to-green-900/30 rounded-xl border border-blue-800/50">
        <div className="h-14 w-14 rounded-full bg-gradient-to-r from-blue-500 to-green-600 flex items-center justify-center mb-4 shadow-lg animate-float">
          <Globe size={28} className="text-white" />
        </div>
        <h2 className="text-2xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-green-400">
          {teamIcon}Community & Collaboration
        </h2>
        <p className="text-gray-300 text-center mb-6 max-w-lg">
          Share your analysis with teammates or create public reports for
          clients and followers
        </p>
      </div>

      <Tabs defaultValue="team" className="w-full">
        <TabsContent value="team" className="animate-fade-in relative">
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm z-10 rounded-lg flex items-center justify-center">
            <div className="text-center space-y-2 p-6 bg-gray-900/80 rounded-lg border border-blue-500/30 max-w-md mx-auto">
              <h3 className="text-xl font-semibold text-blue-400">{clockIcon}Coming Soon</h3>
              <p className="text-gray-300">Team collaboration features are currently in development. Stay tuned for updates!</p>
            </div>
          </div>
          <Card className="glass-card bg-secondary/80 border-gray-700">
            <CardHeader className="flex flex-row items-center justify-between pb-2 border-b border-gray-700/50">
              <div className="flex items-center gap-2">
                <div className="h-9 w-9 rounded-full bg-gradient-to-r from-green-500 to-green-700 flex items-center justify-center">
                  <Users size={18} className="text-white" />
                </div>
                <h3 className="text-lg font-semibold">
                  {teamIcon}Team Accounts
                </h3>
              </div>
              <span className="px-2 py-1 badge-premium bg-gradient-to-r from-yellow-600/50 to-yellow-500/50 text-yellow-200 text-xs rounded-full border border-yellow-500/30">
                PREMIUM
              </span>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="space-y-6">
                <div className="p-5 bg-gray-800/30 rounded-lg border border-gray-700/50">
                  <h4 className="text-center font-medium mb-4">
                    Invite Team Members
                  </h4>
                  <div className="flex items-center space-x-2 p-3 bg-gray-700/30 rounded-lg border border-gray-600/50 mb-4">
                    <input
                      type="email"
                      placeholder="colleague@example.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="flex-1 bg-transparent border-none focus:outline-none text-sm"
                      onKeyPress={(e) => e.key === 'Enter' && handleInviteTeam()}
                    />
                    <Button
                      onClick={handleInviteTeam}
                      variant="outline"
                      size="sm"
                      disabled={isInviting || !email}
                      className="text-xs bg-blue-900/30 hover:bg-blue-800/50 text-blue-300 border-blue-700/30 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isInviting ? "Inviting..." : "Invite"}
                    </Button>
                  </div>
                  <div className="space-y-3">
                    {teamMembers.length === 0 ? (
                      <p className="text-center text-sm text-gray-400">No team members yet</p>
                    ) : (
                      teamMembers.map(member => (
                        <div key={member.id} className="flex justify-between items-center p-3 bg-gray-700/20 rounded-lg border border-gray-600/30">
                          <div className="flex items-center gap-2">
                            <div className={`w-8 h-8 rounded-full bg-gradient-to-br ${member.status === 'active' ? 'from-green-500 to-green-700' : 'from-blue-500 to-blue-700'} flex items-center justify-center`}>
                              <span className="text-xs text-white">{member.initials}</span>
                            </div>
                            <div>
                              <p className="text-sm">{member.name}</p>
                              <p className="text-xs text-gray-400">{member.role}</p>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className={`px-2 py-0.5 text-xs rounded-full border ${member.status === 'active' ? 'bg-green-900/30 text-green-400 border-green-700/30' : 'bg-yellow-900/30 text-yellow-400 border-yellow-700/30'}`}>
                              {member.status === 'active' ? 'Active' : 'Pending'}
                            </span>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="p-0 h-6 w-6 hover:bg-red-900/20"
                              onClick={() => setTeamMembers(prev => prev.filter(m => m.id !== member.id))}
                            >
                              <X size={14} className="text-red-400" />
                            </Button>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>

                <div className="p-4 bg-gray-900/30 rounded-lg border border-yellow-700/30 text-center">
                  <div className="flex items-center justify-center gap-2 mb-2">
                    <Lock size={16} className="text-yellow-400" />
                    <p className="text-yellow-400">Premium Feature</p>
                  </div>
                  <p className="text-sm text-gray-300 mb-4">
                    Upgrade to Premium to invite teammates and collaborate on
                    reports. Team members can comment, edit, and help optimize
                    your content.
                  </p>
                  <Button
                    onClick={handleInviteTeam}
                    className="bg-gradient-to-r from-yellow-700/50 to-yellow-600/50 hover:from-yellow-700/70 hover:to-yellow-600/70 text-yellow-200 border border-yellow-700/30"
                  >
                    Unlock Team Features
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
