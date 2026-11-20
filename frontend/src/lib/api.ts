type HealthResponse = {
  status: string;
  service: string;
};

type GenerateTextResponse = {
  generated_text: string;
};

const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "") ??
  "http://localhost:8000";

export async function getBackendHealth(
  signal?: AbortSignal,
): Promise<HealthResponse> {
  const response = await fetch(`${apiBaseUrl}/health`, { signal });

  if (!response.ok) {
    throw new Error(`Backend health check failed (${response.status})`);
  }

  const payload: unknown = await response.json();

  if (!isHealthResponse(payload)) {
    throw new Error("Backend returned an invalid health response");
  }

  return payload;
}

export async function generateText(
  prompt: string,
  signal?: AbortSignal,
): Promise<string> {
  const options: RequestInit = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      max_new_tokens: 100,
      temperature: 1.0,
    }),
  };
  if (signal) {
    options.signal = signal;
  }

  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}/generate`, options);
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw error;
    }
    throw new Error("Unable to reach the MyLLM backend. Please try again.");
  }

  if (!response.ok) {
    if (response.status === 503) {
      throw new Error("The MyLLM model is unavailable right now.");
    }
    throw new Error(
      `MyLLM text generation failed (${response.status}). Please try again.`,
    );
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error("MyLLM returned an invalid generation response.");
  }

  if (!isGenerateTextResponse(payload)) {
    throw new Error("MyLLM returned an invalid generation response.");
  }

  return payload.generated_text;
}

function isHealthResponse(payload: unknown): payload is HealthResponse {
  return (
    typeof payload === "object" &&
    payload !== null &&
    "status" in payload &&
    typeof payload.status === "string" &&
    "service" in payload &&
    typeof payload.service === "string"
  );
}

function isGenerateTextResponse(
  payload: unknown,
): payload is GenerateTextResponse {
  return (
    typeof payload === "object" &&
    payload !== null &&
    "generated_text" in payload &&
    typeof payload.generated_text === "string"
  );
}
