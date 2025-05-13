
import { ClerkProvider } from "@clerk/clerk-react";
import { dark } from "@clerk/themes";
import { useTheme } from "@/hooks/useTheme";

/**
 * Provides Clerk authentication with a theme based on the current application theme.
 */
export function ClerkProviderWithTheme({ children }: { children: React.ReactNode }) {
  const { theme } = useTheme();
  const publishableKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

  return (
    <ClerkProvider
      publishableKey={publishableKey}
      appearance={{
        baseTheme: theme === "dark" ? dark : undefined,
        variables: { colorPrimary: "#0091FF" },
      }}
    >
      {children}
    </ClerkProvider>
  );
}