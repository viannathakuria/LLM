import { afterEach, describe, expect, it, vi } from "vitest";
import { generateText, getBackendHealth } from "@/lib/api";

describe("getBackendHealth", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe("generateText", () => {
    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it("posts the prompt and generation settings and returns generated text", async () => {
      const controller = new AbortController();
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ generated_text: "Generated response" }),
      });
      vi.stubGlobal("fetch", fetchMock);

      await expect(generateText("Hello", controller.signal)).resolves.toBe(
        "Generated response",
      );
      expect(fetchMock).toHaveBeenCalledWith(
        "http://localhost:8000/generate",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: "Hello",
            max_new_tokens: 100,
            temperature: 1.0,
          }),
          signal: controller.signal,
        },
      );
    });

    it("reports HTTP errors without exposing response details", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: false,
          status: 500,
          json: async () => ({ detail: "private server detail" }),
        }),
      );

      await expect(generateText("Hello")).rejects.toThrow(
        "MyLLM text generation failed (500). Please try again.",
      );
    });

    it("reports when the local model is unavailable", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({ ok: false, status: 503 }),
      );

      await expect(generateText("Hello")).rejects.toThrow(
        "The MyLLM model is unavailable right now.",
      );
    });

    it("rejects malformed or invalid generation response JSON", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          json: async () => {
            throw new SyntaxError("private parse detail");
          },
        }),
      );
      await expect(generateText("Hello")).rejects.toThrow(
        "MyLLM returned an invalid generation response.",
      );

      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          json: async () => ({ generated_text: 42 }),
        }),
      );
      await expect(generateText("Hello")).rejects.toThrow(
        "MyLLM returned an invalid generation response.",
      );
    });

    it("reports an unreachable backend without exposing fetch errors", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockRejectedValue(new TypeError("private network detail")),
      );

      await expect(generateText("Hello")).rejects.toThrow(
        "Unable to reach the MyLLM backend. Please try again.",
      );
    });
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
