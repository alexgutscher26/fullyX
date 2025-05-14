import { Suspense } from 'react';

/**
 * Renders a dashboard page with a loading fallback.
 */
export default function DashboardPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">Dashboard</h1>
      <Suspense fallback={<div>Loading...</div>}>
        {/* Dashboard content will be migrated here */}
      </Suspense>
    </div>
  );
}