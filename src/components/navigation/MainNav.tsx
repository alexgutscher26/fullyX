import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { useAuth, UserButton } from "@clerk/clerk-react";
export default function MainNav() {
  const location = useLocation();
  const [scrolled, setScrolled] = useState(false);

  // Handle scroll effects
  useEffect(() => {
    const handleScroll = () => {
      const isScrolled = window.scrollY > 10;
      if (isScrolled !== scrolled) {
        setScrolled(isScrolled);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, [scrolled]);

  const { isSignedIn } = useAuth();

  // Navigation items with their data
  const navItems = isSignedIn
    ? [
        { to: "/", label: "Home" },
        { to: "/dashboard", label: "Dashboard" },
      ]
    : [
        { to: "/", label: "Home" },
        { to: "/signin", label: "Sign In" },
        { to: "/signup", label: "Sign Up" },
      ];

  // Check if a path is active
  const isActive = (path) => {
    if (path === "/") return location.pathname === "/";
    return location.pathname.startsWith(path);
  };

  return (
    <nav 
      className={`fixed top-4 left-1/2 -translate-x-1/2 z-50 w-[95%] max-w-[1248px] rounded-lg 
      ${scrolled ? "bg-background/95" : "bg-background/80"} 
      backdrop-blur-md border border-border shadow-lg 
      transition-all duration-300 ease-in-out`}
    >
      <div className="flex h-14 items-center justify-between px-4">
        <Link to="/" className="flex items-center space-x-2">
          <span className="hidden font-bold sm:inline-block">
            Fully
          </span>
        </Link>

        <div className="flex items-center space-x-1 md:space-x-2">
          {navItems.map((item) => (
            <Link to={item.to} key={item.to} className="relative group">
              <Button 
                variant={isActive(item.to) ? "default" : "ghost"} 
                size="sm" 
                className={`px-4 transition-all 
                ${isActive(item.to) ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}
              >
                {item.label}
              </Button>
            </Link>
          ))}
          {isSignedIn && (
            <div className="ml-4">
              <UserButton afterSignOutUrl="/" />
            </div>
          )}
        </div>
      </div>
    </nav>
  );
}