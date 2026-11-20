"use client";

import { useEffect, useState } from "react";
import { getBackendHealth } from "@/lib/api";

type Status = "checking" | "connected" | "unavailable";

export function BackendStatus() {
  const [status, setStatus] = useState<Status>("checking");

  useEffect(() => {
    const controller = new AbortController();

    getBackendHealth(controller.signal)
      .then(() => setStatus("connected"))
      .catch(() => {
        if (!controller.signal.aborted) {
          setStatus("unavailable");
        }
      });

    return () => controller.abort();
  }, []);

  const label = {
    checking: "Checking backend...",
    connected: "Backend connected",
    unavailable: "Backend unavailable",
  }[status];

  return (
    <p
      aria-live="polite"
      className="text-sm text-slate-500"
      data-testid="backend-status"
    >
      {label}
    </p>
  );
}
