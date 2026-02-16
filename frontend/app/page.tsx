"use client";

import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Shimmer } from "@/components/ai-elements/shimmer";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputBody,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
  type PromptInputMessage,
} from "@/components/ai-elements/prompt-input";
import { PromptInputProvider } from "@/components/ai-elements/prompt-input";
import { Briefcase, MessageSquare } from "lucide-react";
import { useChat } from "@ai-sdk/react";
import { defineCatalog } from "@json-render/core";
import { JSONUIProvider, defineRegistry, Renderer, schema } from "@json-render/react";
import { z } from "zod";

type CandidateProfile = {
  name: string;
  title?: string;
  location?: string;
  summary?: string;
  matchScore?: string;
  skills: string[];
  languages: string[];
};

const candidateCardCatalog = defineCatalog(schema, {
  components: {
    CardGrid: {
      props: z.object({}),
      slots: ["default"],
    },
    ProfileCard: {
      props: z.object({
        name: z.string(),
        title: z.string().optional(),
        location: z.string().optional(),
        summary: z.string().optional(),
        matchScore: z.string().optional(),
      }),
      slots: ["default"],
    },
    Tag: {
      props: z.object({
        label: z.string(),
      }),
    },
  },
  actions: {},
});

const { registry: candidateCardRegistry } = defineRegistry(candidateCardCatalog, {
  components: {
    CardGrid: ({ children }) => (
      <div className="grid gap-3 sm:grid-cols-2">{children}</div>
    ),
    ProfileCard: ({ props, children }) => (
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <div className="space-y-1">
          <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100">
            {props.name}
          </h3>
          {props.title ? (
            <p className="text-sm text-slate-600 dark:text-slate-300">
              {props.title}
            </p>
          ) : null}
          {props.location ? (
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {props.location}
            </p>
          ) : null}
          {props.matchScore ? (
            <p className="text-xs font-medium text-emerald-600 dark:text-emerald-400">
              Match Score: {props.matchScore}
            </p>
          ) : null}
        </div>
        {props.summary ? (
          <p className="mt-3 text-sm text-slate-700 dark:text-slate-200">
            {props.summary}
          </p>
        ) : null}
        <div className="mt-3 flex flex-wrap gap-2">{children}</div>
      </div>
    ),
    Tag: ({ props }) => (
      <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs text-slate-700 dark:bg-slate-800 dark:text-slate-200">
        {props.label}
      </span>
    ),
  },
});

function CandidateCardsShimmer() {
  return (
    <div className="space-y-3">
      <Shimmer className="text-sm text-slate-500 dark:text-slate-400">
        Finding best-fit candidates...
      </Shimmer>
      <div className="grid gap-3 sm:grid-cols-2">
        {Array.from({ length: 4 }).map((_, index) => (
          <div
            key={`candidate-shimmer-${index}`}
            className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900"
          >
            <div className="space-y-2">
              <div className="h-4 w-2/3 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />
              <div className="h-3 w-1/2 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />
              <div className="h-3 w-1/3 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />
              <div className="h-3 w-full animate-pulse rounded bg-slate-200 dark:bg-slate-700" />
            </div>
            <div className="mt-3 flex gap-2">
              <div className="h-6 w-16 animate-pulse rounded-full bg-slate-200 dark:bg-slate-700" />
              <div className="h-6 w-20 animate-pulse rounded-full bg-slate-200 dark:bg-slate-700" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function normalizeStringArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .map((item) => String(item).trim())
      .filter((item) => item.length > 0);
  }

  if (typeof value === "string") {
    return value
      .split(/[;,|]/g)
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
  }

  return [];
}

function extractJsonObjectPayloads(text: string): string[] {
  const payloads: string[] = [];
  let depth = 0;
  let inString = false;
  let escaping = false;
  let objectStart = -1;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];

    if (inString) {
      if (escaping) {
        escaping = false;
      } else if (char === "\\") {
        escaping = true;
      } else if (char === '"') {
        inString = false;
      }
      continue;
    }

    if (char === '"') {
      inString = true;
      continue;
    }

    if (char === "{") {
      if (depth === 0) {
        objectStart = i;
      }
      depth += 1;
      continue;
    }

    if (char === "}" && depth > 0) {
      depth -= 1;
      if (depth === 0 && objectStart !== -1) {
        payloads.push(text.slice(objectStart, i + 1));
        objectStart = -1;
      }
    }
  }

  return payloads;
}

function looksLikeCandidateJson(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed) return false;

  return (
    trimmed.startsWith("{") ||
    trimmed.startsWith("[") ||
    trimmed.includes('"candidates"') ||
    trimmed.includes('"match_score"') ||
    trimmed.includes('"skills"')
  );
}

function extractCandidatesFromValue(value: unknown): CandidateProfile[] {
  const records =
    Array.isArray(value) && value.every((item) => item && typeof item === "object")
      ? (value as Record<string, unknown>[])
      : value && typeof value === "object"
        ? ([
            (value as Record<string, unknown>).candidates,
            (value as Record<string, unknown>).profiles,
            (value as Record<string, unknown>).results,
            (value as Record<string, unknown>).matches,
          ].find(
            (item) =>
              Array.isArray(item) &&
              item.every((candidate) => candidate && typeof candidate === "object")
          ) as Record<string, unknown>[] | undefined) ?? []
        : [];

  return records
    .map((candidate) => {
      const name =
        String(
          candidate.name ??
            candidate.full_name ??
            candidate.candidate_name ??
            candidate.candidate ??
            ""
        ).trim() || "Candidate";

      const title = String(candidate.title ?? candidate.role ?? "").trim();
      const location = String(candidate.location ?? candidate.city ?? "").trim();
      const summary = String(
        candidate.summary ?? candidate.bio ?? candidate.profile ?? ""
      ).trim();
      const rawMatch = candidate.match_score ?? candidate.score ?? candidate.match;
      const matchScore =
        rawMatch === undefined || rawMatch === null
          ? ""
          : String(rawMatch).trim();

      return {
        name,
        title: title || undefined,
        location: location || undefined,
        summary: summary || undefined,
        matchScore: matchScore || undefined,
        skills: normalizeStringArray(candidate.skills),
        languages: normalizeStringArray(candidate.languages),
      };
    })
    .filter(
      (candidate) =>
        candidate.name ||
        candidate.title ||
        candidate.summary ||
        candidate.skills.length > 0
    );
}

function parseCandidateProfiles(text: string): CandidateProfile[] | null {
  const payloads: string[] = [];
  const trimmed = text.trim();

  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    payloads.push(trimmed);
  }

  const blockRegex = /```(?:json)?\s*([\s\S]*?)```/gi;
  for (const match of trimmed.matchAll(blockRegex)) {
    if (match[1]) {
      payloads.push(match[1].trim());
    }
  }

  payloads.push(...extractJsonObjectPayloads(trimmed));

  let latestCandidates: CandidateProfile[] | null = null;
  for (const payload of payloads) {
    try {
      const parsed = JSON.parse(payload);
      const candidates = extractCandidatesFromValue(parsed);
      if (candidates.length > 0) {
        latestCandidates = candidates;
      }
    } catch {
      // Ignore invalid JSON payloads and continue fallback rendering.
    }
  }

  return latestCandidates;
}

function buildCandidateCardsSpec(candidates: CandidateProfile[]) {
  const rootId = "candidate-grid";
  const elements: Record<
    string,
    {
      type: string;
      props: Record<string, unknown>;
      children?: string[];
    }
  > = {
    [rootId]: {
      type: "CardGrid",
      props: {},
      children: [],
    },
  };

  const rootChildren: string[] = [];

  candidates.forEach((candidate, index) => {
    const profileId = `candidate-${index}`;
    const tagIds: string[] = [];
    const tags = [...candidate.skills, ...candidate.languages]
      .map((item) => item.trim())
      .filter((item) => item.length > 0);

    tags.forEach((tag, tagIndex) => {
      const tagId = `${profileId}-tag-${tagIndex}`;
      tagIds.push(tagId);
      elements[tagId] = {
        type: "Tag",
        props: { label: tag },
      };
    });

    elements[profileId] = {
      type: "ProfileCard",
      props: {
        name: candidate.name,
        title: candidate.title,
        location: candidate.location,
        summary: candidate.summary,
        matchScore: candidate.matchScore,
      },
      children: tagIds,
    };
    rootChildren.push(profileId);
  });

  elements[rootId].children = rootChildren;

  return {
    root: rootId,
    elements,
  };
}

export default function Home() {
  const { messages, sendMessage, status } = useChat();
  const latestMessageId = messages[messages.length - 1]?.id;

  const handleSubmit = (message: PromptInputMessage) => {
    if (message.text?.trim()) {
      sendMessage({ text: message.text });
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-linear-to-b from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      <header className="border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center gap-3">
          <div className="p-2 rounded-lg bg-emerald-100 dark:bg-emerald-900/30">
            <Briefcase className="size-6 text-emerald-600 dark:text-emerald-400" />
          </div>
          <div>
            <h1 className="font-semibold text-lg text-slate-900 dark:text-slate-100">
              Youth Opportunity Matcher
            </h1>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Find talent or opportunities with AI-powered matching
            </p>
          </div>
        </div>
      </header>

      <main className="flex-1 flex flex-col max-w-4xl w-full mx-auto p-4">
        <PromptInputProvider initialInput="">
          <div className="flex flex-col h-[calc(100vh-12rem)] rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-lg overflow-hidden">
            <div className="flex flex-col flex-1 min-h-0">
              <Conversation className="flex-1">
                <ConversationContent>
                  {messages.length === 0 ? (
                    <ConversationEmptyState
                      icon={
                        <MessageSquare className="size-12 text-slate-400" />
                      }
                      title="Start matching"
                      description="Ask to find youth candidates for a role, post a job requirement, or get matching statistics. Try: &quot;Find me someone who speaks English and has Python skills&quot;"
                    />
                  ) : (
                    messages.map((message) => (
                      <Message from={message.role} key={message.id}>
                        <MessageContent>
                          {message.parts.map((part, i) => {
                            switch (part.type) {
                              case "text":
                                const profiles = parseCandidateProfiles(part.text);
                                const isStreamingAssistantResponse =
                                  status === "streaming" &&
                                  message.role === "assistant" &&
                                  message.id === latestMessageId;
                                const shouldHideRawJson =
                                  message.role === "assistant" &&
                                  looksLikeCandidateJson(part.text);

                                if (profiles && profiles.length > 0) {
                                  return (
                                    <div
                                      key={`${message.id}-${i}`}
                                      className="w-full"
                                    >
                                      <JSONUIProvider
                                        registry={candidateCardRegistry}
                                      >
                                        <Renderer
                                          spec={buildCandidateCardsSpec(profiles)}
                                          registry={candidateCardRegistry}
                                        />
                                      </JSONUIProvider>
                                    </div>
                                  );
                                }

                                if (
                                  isStreamingAssistantResponse &&
                                  shouldHideRawJson
                                ) {
                                  return (
                                    <div
                                      key={`${message.id}-${i}`}
                                      className="w-full"
                                    >
                                      <CandidateCardsShimmer />
                                    </div>
                                  );
                                }

                                if (shouldHideRawJson) {
                                  return (
                                    <div
                                      key={`${message.id}-${i}`}
                                      className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600 dark:border-slate-700 dark:bg-slate-800/40 dark:text-slate-300"
                                    >
                                      Candidate data received but could not be
                                      rendered as cards. Please retry.
                                    </div>
                                  );
                                }

                                return (
                                  <MessageResponse
                                    key={`${message.id}-${i}`}
                                    className="prose dark:prose-invert prose-sm max-w-none"
                                  >
                                    {part.text}
                                  </MessageResponse>
                                );
                              default:
                                return null;
                            }
                          })}
                        </MessageContent>
                      </Message>
                    ))
                  )}
                </ConversationContent>
                <ConversationScrollButton />
              </Conversation>

              <div className="p-4 border-t border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50">
                <PromptInput
                  onSubmit={handleSubmit}
                  className="w-full max-w-2xl mx-auto"
                >
                  <PromptInputBody>
                    <PromptInputTextarea
                      placeholder="Find candidates, post jobs, or ask about matching..."
                      className="pr-12 min-h-12"
                    />
                  </PromptInputBody>
                  <PromptInputFooter>
                    <PromptInputTools />
                    <PromptInputSubmit
                      status={status === "streaming" ? "streaming" : "ready"}
                    />
                  </PromptInputFooter>
                </PromptInput>
              </div>
            </div>
          </div>
        </PromptInputProvider>
      </main>
    </div>
  );
}
