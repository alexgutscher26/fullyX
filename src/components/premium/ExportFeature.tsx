import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { FileText, Download, Check, FileSpreadsheet } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "@/components/ui/sonner";

interface ExportFeatureProps {
  includeEmojis?: boolean;
}

export function ExportFeature({ includeEmojis = true }: ExportFeatureProps) {
  // Emoji or text representations based on toggle
  const exportIcon = includeEmojis ? "📤 " : "";
  const pdfIcon = includeEmojis ? "📑 " : "";
  const csvIcon = includeEmojis ? "📊 " : "";
  
  const handleExportPDF = () => {
    toast.info("Premium Feature", {
      description: "Upgrade to Premium to export as PDF",
      action: {
        label: "Upgrade",
        onClick: () => toast("Redirecting to upgrade page...")
      },
    });
  };
  
  const handleExportCSV = () => {
    toast.info("Premium Feature", {
      description: "Upgrade to Premium to export as CSV",
      action: {
        label: "Upgrade",
        onClick: () => toast("Redirecting to upgrade page...")
      },
    });
  };
  
  return (
    <Card className="glass-card card-hover">
      <CardHeader className="flex flex-row items-center justify-between pb-2 border-b border-gray-700/50">
        <div className="flex items-center gap-2">
          <div className="feature-icon-container w-8 h-8">
            <FileText size={15} className="text-white" />
          </div>
          <h3 className="text-lg font-semibold">{exportIcon}Export Analysis</h3>
        </div>
        <span className="px-2 py-1 badge-premium text-xs rounded-full">
          PREMIUM
        </span>
      </CardHeader>
      <CardContent className="pt-6">
        <div className="space-y-6">
          <div className="p-5 bg-gray-800/30 rounded-lg border border-gray-700/50 text-center">
            <p className="text-gray-300 mb-6">
              Export your analysis results in different formats for reporting, presentations, or further analysis.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="p-5 bg-gray-800/50 rounded-lg border border-gray-700/30 flex flex-col items-center space-y-4">
                <div className="h-12 w-12 rounded-full bg-gradient-to-br from-red-500/30 to-red-700/30 flex items-center justify-center border border-red-500/30">
                  <FileText size={24} className="text-red-400" />
                </div>
                <h4 className="font-medium">{pdfIcon}PDF Report</h4>
                <p className="text-xs text-gray-400 text-center">
                  Complete analysis with visual charts and recommendations
                </p>
                <Button 
                  onClick={handleExportPDF}
                  className="w-full flex items-center gap-2 bg-red-900/50 hover:bg-red-800/70 text-red-200 border border-red-700/30"
                >
                  <Download size={16} />
                  Export as PDF
                </Button>
              </div>
              
              <div className="p-5 bg-gray-800/50 rounded-lg border border-gray-700/30 flex flex-col items-center space-y-4">
                <div className="h-12 w-12 rounded-full bg-gradient-to-br from-green-500/30 to-green-700/30 flex items-center justify-center border border-green-500/30">
                  <FileSpreadsheet size={24} className="text-green-400" />
                </div>
                <h4 className="font-medium">{csvIcon}CSV Data</h4>
                <p className="text-xs text-gray-400 text-center">
                  Raw data export for spreadsheet analysis and custom reporting
                </p>
                <Button 
                  onClick={handleExportCSV}
                  className="w-full flex items-center gap-2 bg-green-900/50 hover:bg-green-800/70 text-green-200 border border-green-700/30"
                >
                  <Download size={16} />
                  Export as CSV
                </Button>
              </div>
            </div>
          </div>
          
          <div className="p-4 bg-gray-900/30 rounded-lg border border-yellow-700/30 text-center">
            <div className="flex items-center justify-center gap-2 mb-2">
              <Check size={16} className="text-yellow-400" />
              <p className="text-yellow-400">Premium Feature</p>
            </div>
            <p className="text-sm text-gray-300">
              Upgrade to Premium to unlock data export functionality for your reports
            </p>
            <Button 
              className="mt-4 bg-gradient-to-r from-yellow-700/50 to-yellow-600/50 hover:from-yellow-700/70 hover:to-yellow-600/70 text-yellow-200 border border-yellow-700/30"
              size="sm"
              onClick={handleExportPDF}
            >
              Unlock Export Features
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
