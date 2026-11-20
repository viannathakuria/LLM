"use client";

import { FormEvent, useRef, useState } from "react";
import { BackendStatus } from "@/components/BackendStatus";
import { createPlaceholderResponse } from "@/lib/chat";

type Message = {
  id: number;
  role: "assistant" | "user";
  text: string;
};

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 0,
      role: "assistant",
      text: "Welcome to MyLLM. This is a placeholder, not a model-generated response.",
    },
  ]);
  const [draft, setDraft] = useState("");
  const nextMessageId = useRef(1);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = draft.trim();

    if (!text) {
      return;
    }

    const messageId = nextMessageId.current;
    nextMessageId.current += 2;
    setMessages((current) => [
      ...current,
      { id: messageId, role: "user", text },
      {
        id: messageId + 1,
        role: "assistant",
        text: createPlaceholderResponse(),
      },
    ]);
    setDraft("");
  }

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-4xl flex-col px-4 py-8 sm:px-8">
      <header className="mb-8 flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">MyLLM</h1>
          <p className="mt-1 text-sm text-slate-500">
            An independent language model project
          </p>
        </div>
        <BackendStatus />
      </header>

      <section
        aria-label="Conversation"
        aria-live="polite"
        className="flex-1 space-y-4 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6"
      >
        <p className="rounded-lg bg-amber-50 px-4 py-3 text-sm text-amber-900">
          Demo only: responses are fixed placeholders. No AI model or external AI
          service is connected.
        </p>
        {messages.map((message) => (
          <article
            className={`max-w-[85%] rounded-2xl px-4 py-3 ${
              message.role === "user"
                ? "ml-auto bg-slate-900 text-white"
                : "bg-slate-100 text-slate-800"
            }`}
            key={message.id}
          >
            <p className="mb-1 text-xs font-semibold uppercase tracking-wide opacity-70">
              {message.role === "user" ? "You" : "MyLLM placeholder"}
            </p>
            <p className="whitespace-pre-wrap text-sm leading-6">
              {message.text}
            </p>
          </article>
        ))}
      </section>

      <form className="mt-4 flex gap-2" onSubmit={handleSubmit}>
        <label className="sr-only" htmlFor="chat-message">
          Message
        </label>
        <input
          autoComplete="off"
          className="min-w-0 flex-1 rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm outline-none transition focus:border-slate-500 focus:ring-2 focus:ring-slate-200"
          id="chat-message"
          onChange={(event) => setDraft(event.target.value)}
          placeholder="Type a message..."
          value={draft}
        />
        <button
          className="rounded-xl bg-slate-900 px-5 py-3 text-sm font-medium text-white transition hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
          disabled={!draft.trim()}
          type="submit"
        >
          Send
        </button>
      </form>
    </main>
  );
}
