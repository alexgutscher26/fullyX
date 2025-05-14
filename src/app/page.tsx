import Link from 'next/link';
import { Button } from "@/components/ui/button";
import MainNav from "@/components/navigation/MainNav";
import "@/styles/animations.css";

/**
 * Renders background elements with grid overlay and animated circle.
 */
const BackgroundElements = () => (
  <>
    <div className="absolute inset-0 bg-grid-white/[0.02] bg-[size:75px_75px] -z-10" />
    <div className="absolute inset-0 flex items-center justify-center -z-10">
      <div className="w-[800px] h-[800px] bg-primary/30 rounded-full blur-3xl opacity-20 animate-pulse" />
    </div>
  </>
);

/**
 * Renders a hero section with a title, description, and call-to-action button.
 */
const HeroSection = () => (
  <div className="max-w-3xl mx-auto text-center space-y-8">
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary/70">
        Supercharge Your Social Media Growth
      </h1>
      <p className="text-lg sm:text-xl text-muted-foreground/90 max-w-2xl mx-auto leading-relaxed">
        Analyze, optimize, and enhance your social media posts with AI-powered insights.
        Get better engagement, reach, and results.
      </p>
    </div>
    <CallToAction />
  </div>
);

/**
 * Renders a call-to-action button that links to the dashboard.
 */
const CallToAction = () => (
  <div className="flex justify-center gap-4 pt-4 animate-fade-in [animation-delay:200ms]">
    <Link href="/dashboard">
      <Button 
        size="lg" 
        className="gap-2 bg-primary/90 hover:bg-primary shadow-lg hover:shadow-xl transition-all duration-300 group"
      >
        Get Started
        <span 
          aria-hidden="true" 
          className="group-hover:translate-x-1 transition-transform duration-300"
        >
          →
        </span>
      </Button>
    </Link>
  </div>
);

/**
 * Renders the main page component with a background gradient, header, and hero section.
 */
export default function Page() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background via-background/95 to-background/90 relative overflow-hidden">
      {/* Background elements */}
      <BackgroundElements />
      
      {/* Header with navigation */}
      <header className="container mx-auto px-4 py-6">
        <MainNav />
      </header>
      
      {/* Main content */}
      <main className="container mx-auto px-4 py-24 sm:py-32">
        <HeroSection />
      </main>
    </div>
  );
}