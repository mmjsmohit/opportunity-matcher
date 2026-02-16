import {
  createUIMessageStream,
  createUIMessageStreamResponse,
  generateId,
  type UIMessage,
} from "ai";

export const maxDuration = 60;

const AGNO_API_URL =
  process.env.AGNO_API_URL || "http://localhost:7777";
const TEAM_ID = "youth-matching-platform";

function extractDeltaFromAgnoEvent(payload: unknown): string {
  if (typeof payload === "string") {
    return payload;
  }

  if (!payload || typeof payload !== "object") {
    return "";
  }

  const record = payload as Record<string, unknown>;

  if (typeof record.content === "string") return record.content;
  if (typeof record.delta === "string") return record.delta;
  if (typeof record.text === "string") return record.text;

  const nested = record.data;
  if (nested && typeof nested === "object") {
    const nestedRecord = nested as Record<string, unknown>;
    if (typeof nestedRecord.content === "string") return nestedRecord.content;
    if (typeof nestedRecord.delta === "string") return nestedRecord.delta;
    if (typeof nestedRecord.text === "string") return nestedRecord.text;
  }

  return "";
}

function getLastUserMessageText(messages: UIMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === "user") {
      const textPart = msg.parts?.find((p) => p.type === "text");
      return textPart && "text" in textPart ? textPart.text : "";
    }
  }
  return "";
}

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const messages: UIMessage[] = body.messages ?? [];
    const sessionId = body.sessionId ?? generateId();

    const lastMessage = getLastUserMessageText(messages);
    if (!lastMessage.trim()) {
      return new Response(
        JSON.stringify({ error: "No message content provided" }),
        { status: 400 }
      );
    }

    const stream = createUIMessageStream({
      originalMessages: messages,
      async execute({ writer }) {
        const textId = generateId();
        let accumulatedText = "";

        try {
          const formData = new FormData();
          formData.append("message", lastMessage);
          formData.append("session_id", sessionId);
          formData.append("stream", "true");

          const response = await fetch(
            `${AGNO_API_URL}/teams/${TEAM_ID}/runs`,
            {
              method: "POST",
              body: formData,
            }
          );

          if (!response.ok) {
            const errText = await response.text();
            throw new Error(`Agno API error: ${response.status} - ${errText}`);
          }

          const contentType = response.headers.get("content-type") || "";
          let started = false;

          const emitDelta = (rawDelta: string) => {
            let delta = rawDelta;
            if (!delta) return;

            // Filter Agno tool-debug lines that are not user-facing content.
            if (
              /^[a-zA-Z_][a-zA-Z0-9_]*\(.*\)\s+completed in\s+\d+(\.\d+)?s\.\s*$/.test(
                delta
              )
            ) {
              return;
            }

            // Some Agno events send full snapshots instead of strict deltas.
            // Convert snapshots to unseen suffix to avoid duplicate output.
            if (accumulatedText && delta.startsWith(accumulatedText)) {
              delta = delta.slice(accumulatedText.length);
            }
            if (accumulatedText && delta.includes(accumulatedText)) {
              const idx = delta.lastIndexOf(accumulatedText);
              if (idx !== -1) {
                delta = delta.slice(idx + accumulatedText.length);
              }
            }

            // Skip exact duplicates and repeated large chunks.
            if (!delta) return;
            if (delta === accumulatedText) return;
            if (accumulatedText.endsWith(delta)) return;
            if (delta.length > 200 && accumulatedText.includes(delta)) return;

            if (!started) {
              writer.write({ type: "text-start", id: textId });
              started = true;
            }
            writer.write({ type: "text-delta", id: textId, delta });
            accumulatedText += delta;
          };

          if (contentType.includes("text/event-stream") && response.body) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let shouldStop = false;

            while (!shouldStop) {
              const { done, value } = await reader.read();
              if (done) break;

              buffer += decoder.decode(value, { stream: true });

              let newlineIndex = buffer.indexOf("\n");
              while (newlineIndex !== -1) {
                const line = buffer.slice(0, newlineIndex).trimEnd();
                buffer = buffer.slice(newlineIndex + 1);
                newlineIndex = buffer.indexOf("\n");

                if (!line || line.startsWith(":")) continue;
                if (!line.startsWith("data:")) continue;

                const raw = line.slice(5).trim();
                if (!raw) continue;
                if (raw === "[DONE]") {
                  shouldStop = true;
                  break;
                }

                let parsed: unknown = raw;
                try {
                  parsed = JSON.parse(raw);
                } catch {
                  // Keep raw text payload when not JSON
                }

                emitDelta(extractDeltaFromAgnoEvent(parsed));
              }
            }
          } else {
            // Fallback for non-streaming responses
            const data = await response.json();
            const content =
              String(
                data.content ??
                  data.output?.content ??
                  data.output ??
                  (typeof data === "string" ? data : "")
              ) || "No response received.";
            emitDelta(content);
          }

          if (started) {
            writer.write({ type: "text-end", id: textId });
          } else {
            writer.write({
              type: "error",
              errorText: "No stream content received from backend",
            });
          }
        } catch (error) {
          writer.write({
            type: "error",
            errorText:
              error instanceof Error ? error.message : "An error occurred",
          });
        }
      },
    });

    return createUIMessageStreamResponse({ stream });
  } catch (error) {
    console.error("Chat API error:", error);
    return new Response(
      JSON.stringify({
        error: error instanceof Error ? error.message : "Internal server error",
      }),
      { status: 500 }
    );
  }
}
