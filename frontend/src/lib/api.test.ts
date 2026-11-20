import { afterEach, describe, expect, it, vi } from "vitest";
import { getBackendHealth } from "@/lib/api";

describe("getBackendHealth", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns the backend health payload", async () => {
    const payload = { status: "ok", service: "myllm-backend" };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => payload,
      }),
    );

    await expect(getBackendHealth()).resolves.toEqual(payload);
  });

  it("reports an unsuccessful health response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 503 }),
    );

    await expect(getBackendHealth()).rejects.toThrow(
      "Backend health check failed (503)",
    );
  });

  it("reports an invalid health payload", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ status: "ok" }),
      }),
    );

    await expect(getBackendHealth()).rejects.toThrow(
      "Backend returned an invalid health response",
    );
  });
});
