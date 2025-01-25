import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "./ui/button";

interface MetricsPanelToggleProps {
  isOpen: boolean;
  onToggle: () => void;
}

export function MetricsPanelToggle({
  isOpen,
  onToggle,
}: MetricsPanelToggleProps) {
  return (
    <div className="toggle-sidebar">
      <Button
        variant="outline"
        size="icon"
        className="w-full h-full rounded-lg z-10 grow-0"
        onClick={onToggle}
      >
        {isOpen ? <ChevronRight /> : <ChevronLeft />}
      </Button>
      <style jsx>{`
        .toggle-sidebar {
          grid-area: toggle-sidebar;
        }
      `}</style>
    </div>
  );
}
