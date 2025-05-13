import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import MainNav from "@/components/navigation/MainNav";

/**
 * A functional component representing the landing page of an application.
 */
const Landing = () => {
  return (
    <div className="min-h-screen bg-background">
      <header className="container mx-auto px-4 py-6 flex justify-between items-center">
        <MainNav />
      </header>
      <main className="container mx-auto px-4 py-16">
        <div className="max-w-3xl mx-auto text-center space-y-8">
          <h1 className="text-5xl font-bold tracking-tight">
            Supercharge Your Social Media Growth
          </h1>
          <p className="text-xl text-muted-foreground">
            Analyze, optimize, and enhance your social media posts with AI-powered insights.
            Get better engagement, reach, and results.
          </p>
          <div className="flex justify-center gap-4">
            <Link to="/dashboard">
              <Button size="lg" className="gap-2">
                Get Started
                <span aria-hidden="true">→</span>
              </Button>
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Landing;