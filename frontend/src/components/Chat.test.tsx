import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Chat } from "@/components/Chat";

type MockResponse = {
  ok: boolean;
  status: number;
  json: () => Promise<unknown>;
};

function jsonResponse(payload: unknown, status = 200): MockResponse {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => payload,
  };
}

function stubFetch(
  generateResponse: () => Promise<MockResponse>,
): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn((input: RequestInfo | URL) =>
    String(input).endsWith("/health")
      ? Promise.resolve(jsonResponse({ status: "ok", service: "myllm-backend" }))
      : generateResponse(),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

describe("Chat", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("preserves the welcome message and identifies local model replies", () => {
    stubFetch(() => Promise.resolve(jsonResponse({ generated_text: "unused" })));

    render(<Chat />);

    expect(
      screen.getByText(
        "Welcome to MyLLM. This is a placeholder, not a model-generated response.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/configured local MyLLM model/i),
    ).toBeInTheDocument();
  });

  it("shows the user message and loading state, then displays generated text", async () => {
    const response = deferred<MockResponse>();
    const fetchMock = stubFetch(() => response.promise);

    render(<Chat />);
    fireEvent.change(screen.getByLabelText("Message"), {
      target: { value: "  Hello MyLLM  " },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(screen.getByText("Hello MyLLM")).toBeInTheDocument();
    expect(screen.getByText("Generating response...")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();

    response.resolve(jsonResponse({ generated_text: "A generated answer" }));

    expect(await screen.findByText("A generated answer")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Message"), {
      target: { value: "Next message" },
    });
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Send" })).not.toBeDisabled(),
    );
    expect(
      fetchMock.mock.calls.filter(([url]) => String(url).endsWith("/generate")),
    ).toHaveLength(1);
  });

  it("shows a friendly error and preserves the conversation on failure", async () => {
    stubFetch(() =>
      Promise.resolve(jsonResponse({ detail: "private server detail" }, 503)),
    );

    render(<Chat />);
    fireEvent.change(screen.getByLabelText("Message"), {
      target: { value: "Keep my message" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Keep my message")).toBeInTheDocument();
    expect(
      await screen.findByText(
        "Sorry, I couldn't generate a response right now. Please try again.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Welcome to MyLLM. This is a placeholder, not a model-generated response.",
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText(/private server detail/i)).not.toBeInTheDocument();
  });

  it("prevents duplicate submissions while a request is pending", async () => {
    const response = deferred<MockResponse>();
    const fetchMock = stubFetch(() => response.promise);

    render(<Chat />);
    fireEvent.change(screen.getByLabelText("Message"), {
      target: { value: "Only once" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    const form = screen.getByLabelText("Message").closest("form");
    expect(form).not.toBeNull();
    fireEvent.submit(form!);
    fireEvent.submit(form!);

    expect(
      fetchMock.mock.calls.filter(([url]) => String(url).endsWith("/generate")),
    ).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();

    response.resolve(jsonResponse({ generated_text: "Done" }));
    await screen.findByText("Done");
  });

  it("does not submit empty or whitespace-only prompts", () => {
    const fetchMock = stubFetch(() =>
      Promise.resolve(jsonResponse({ generated_text: "unused" })),
    );

    render(<Chat />);
    fireEvent.change(screen.getByLabelText("Message"), {
      target: { value: "   \n\t  " },
    });

    const sendButton = screen.getByRole("button", { name: "Send" });
    expect(sendButton).toBeDisabled();
    const form = screen.getByLabelText("Message").closest("form");
    expect(form).not.toBeNull();
    fireEvent.submit(form!);

    expect(
      fetchMock.mock.calls.filter(([url]) => String(url).endsWith("/generate")),
    ).toHaveLength(0);
  });
});
