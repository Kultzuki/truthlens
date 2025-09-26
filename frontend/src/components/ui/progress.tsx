"use client"

import * as React from "react";
import { cn } from "@/lib/utils";

export interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number;
}

export const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(({ value = 0, className, ...props }, ref) => {
  const pct = Math.min(100, Math.max(0, value));
  return (
    <div ref={ref} className={cn("relative h-2 w-full overflow-hidden rounded bg-muted", className)} {...props}>
      <div className="h-full bg-primary transition-all" style={{ width: `${pct}%` }} />
    </div>
  );
});
Progress.displayName = "Progress";
