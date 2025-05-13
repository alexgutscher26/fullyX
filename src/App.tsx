
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { SignIn, SignUp, useAuth } from "@clerk/clerk-react";
import { ClerkProviderWithTheme } from "@/components/providers/clerk-provider";
import Landing from "./pages/Landing";
import Dashboard from "./pages/Dashboard";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

// Protected route wrapper
/**
 * Renders protected routes, redirecting to sign-in if not authenticated.
 */
const ProtectedRoute = ({ children }) => {
  const { isSignedIn, isLoaded } = useAuth();

  if (!isLoaded) return null;
  if (!isSignedIn) return <Navigate to="/signin" />;

  return children;
};

/**
 * The main application component that provides routing and context providers.
 */
const App = () => (
  <ClerkProviderWithTheme>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/signin" element={<SignIn />} />
            <Route path="/signup" element={<SignUp />} />
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <Dashboard />
                </ProtectedRoute>
              }
            />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  </ClerkProviderWithTheme>
);

export default App;
