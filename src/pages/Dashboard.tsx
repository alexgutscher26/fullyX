import Index from "./Index";

/**
 * Renders the main dashboard component with a background and an index component inside.
 */
const Dashboard = () => {
  return (
    <div className="min-h-screen bg-background">
        <Index />
    </div>
  );
};

export default Dashboard;
