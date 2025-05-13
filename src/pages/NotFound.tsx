import { useLocation, Link } from "react-router-dom";
import { useEffect, useState } from "react";

const NotFound = () => {
  const location = useLocation();
  const [countdown, setCountdown] = useState(5);

  // Log the 404 error with additional details
  useEffect(() => {
    const userAgent = navigator.userAgent;
    const referrer = document.referrer;

    console.error("404 Error encountered:", {
      path: location.pathname,
      timestamp: new Date().toISOString(),
      userAgent,
      referrer: referrer || "direct navigation",
    });
  }, [location.pathname]);

  // Set up automatic redirect countdown
  useEffect(() => {
    if (countdown > 0) {
      const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
      return () => clearTimeout(timer);
    } else {
      window.location.href = "/";
    }
  }, [countdown]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-md w-full p-8 bg-white rounded-lg shadow-lg text-center">
        <div className="mb-6">
          <span className="inline-block text-red-500 text-6xl font-bold">
            404
          </span>
          <div className="w-16 h-1 mx-auto bg-red-500 my-4"></div>
        </div>

        <h1 className="text-2xl font-bold text-gray-800 mb-4">
          Page Not Found
        </h1>

        <p className="text-gray-600 mb-6">
          We couldn't find the page you're looking for. The page may have been
          moved, deleted, or never existed.
        </p>

        <div className="mb-6 text-sm text-gray-500">
          <p>
            Requested path:{" "}
            <code className="bg-gray-100 px-2 py-1 rounded">
              {location.pathname}
            </code>
          </p>
        </div>

        <div className="space-y-4">
          <Link
            to="/"
            className="block w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-md transition-colors"
          >
            Return to Homepage
          </Link>

          <p className="text-sm text-gray-500">
            Automatically redirecting in {countdown} second
            {countdown !== 1 ? "s" : ""}...
          </p>
        </div>
      </div>
    </div>
  );
};

export default NotFound;
