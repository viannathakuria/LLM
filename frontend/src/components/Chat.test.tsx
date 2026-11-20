import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Chat } from "@/components/Chat";

describe("Chat", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("labels the initial content as a placeholder", () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ status: "ok", service: "myllm-backend" }),
      }),
    );

    render(<Chat />);

    expect(
      screen.getByText(/responses are fixed placeholders/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/not a model-generated response/i),
    ).toBeInTheDocument();
  });

  it("shows a placeholder response after submitting a message", () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ status: "ok", service: "myllm-backend" }),
      }),
    );

    render(<Chat />);
    fireEvent.change(screen.getByLabelText("Message"), {
      target: { value: "Hello MyLLM" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(screen.getByText("Hello MyLLM")).toBeInTheDocument();
    expect(
      screen.getByText(/does not have a language model connected yet/i),
    ).toBeInTheDocument();
  });
});
