import Anthropic from "@anthropic-ai/sdk";
import { config } from "./config.js";

const client = new Anthropic();

/** Un messaggio della conversazione come lo invia il client dell'API. */
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface Source {
  url: string;
  title: string | null;
}

/** Eventi emessi durante la generazione (usati per lo streaming SSE). */
export type AssistantEvent =
  | { type: "text"; text: string }
  | { type: "search"; query: string }
  | { type: "fetch"; url: string }
  | { type: "source"; source: Source };

export interface AssistantResult {
  answer: string;
  sources: Source[];
  stopReason: string | null;
  model: string;
  usage: { inputTokens: number; outputTokens: number; webSearches: number };
}

function systemPrompt(): string {
  return `Sei ${config.assistantName}, l'assistente AI generalista di Ticly.

Il tuo compito è rispondere a domande di ogni tipo in modo chiaro, accurato e utile, cercando informazioni aggiornate sul web quando serve.

Linee guida:
- Rispondi nella lingua dell'utente (di default italiano).
- Usa la ricerca web (web_search) per fatti recenti, notizie, prezzi, orari, dati che cambiano nel tempo o qualsiasi cosa di cui non sei sicuro. Per domande generali o di ragionamento rispondi direttamente senza cercare.
- Quando l'utente ti dà un link o vuoi leggere una pagina per intero, usa web_fetch.
- Basa le affermazioni sulle fonti trovate e non inventare dati: se le fonti non bastano o si contraddicono, dillo.
- Vai dritto al punto: prima la risposta, poi i dettagli utili. Usa Markdown (elenchi, tabelle) solo quando aiuta la lettura.
- Non sei Gemini, ChatGPT o Claude: sei ${config.assistantName} di Ticly. Se ti chiedono su che tecnologia sei basato, puoi dire che usi un modello di Anthropic.

Data di oggi: ${new Date().toISOString().slice(0, 10)}.`;
}

function tools(): Anthropic.Beta.BetaToolUnion[] {
  return [
    { type: "web_search_20260209", name: "web_search", max_uses: config.maxSearches },
    {
      type: "web_fetch_20260209",
      name: "web_fetch",
      max_uses: config.maxFetches,
      citations: { enabled: true },
    },
  ];
}

/**
 * Esegue una risposta dell'assistente sulla conversazione data.
 * `onEvent` riceve testo e attività di ricerca man mano che arrivano.
 */
export async function runAssistant(
  history: ChatMessage[],
  onEvent: (event: AssistantEvent) => void = () => {},
  signal?: AbortSignal,
): Promise<AssistantResult> {
  const messages: Anthropic.Beta.BetaMessageParam[] = history.map((m) => ({
    role: m.role,
    content: m.content,
  }));

  const sources = new Map<string, Source>();
  const addSource = (url: string, title: string | null) => {
    if (!url || sources.has(url)) return;
    const source = { url, title };
    sources.set(url, source);
    onEvent({ type: "source", source });
  };

  let answer = "";
  let inputTokens = 0;
  let outputTokens = 0;
  let webSearches = 0;
  let final: Anthropic.Beta.BetaMessage | undefined;

  for (let attempt = 0; attempt <= config.maxContinuations; attempt++) {
    const stream = client.beta.messages.stream(
      {
        model: config.model,
        max_tokens: 64000,
        betas: ["server-side-fallback-2026-07-01"],
        fallbacks: "default",
        output_config: { effort: config.effort },
        system: systemPrompt(),
        cache_control: { type: "ephemeral" },
        tools: tools(),
        messages,
      },
      { signal },
    );

    stream.on("text", (delta) => {
      answer += delta;
      onEvent({ type: "text", text: delta });
    });

    stream.on("contentBlock", (block) => {
      if (block.type === "server_tool_use") {
        const input = block.input as { query?: unknown; url?: unknown };
        if (block.name === "web_search" && typeof input.query === "string") {
          onEvent({ type: "search", query: input.query });
        } else if (block.name === "web_fetch" && typeof input.url === "string") {
          onEvent({ type: "fetch", url: input.url });
        }
      } else if (block.type === "web_search_tool_result") {
        // In caso di errore `content` è un oggetto, altrimenti una lista.
        if (Array.isArray(block.content)) {
          for (const r of block.content) addSource(r.url, r.title);
        }
      } else if (block.type === "web_fetch_tool_result") {
        if (block.content.type === "web_fetch_result") {
          addSource(block.content.url, block.content.content.title ?? null);
        }
      } else if (block.type === "text" && block.citations) {
        for (const c of block.citations) {
          if (c.type === "web_search_result_location") addSource(c.url, c.title);
        }
      }
    });

    final = await stream.finalMessage();
    inputTokens += final.usage.input_tokens;
    outputTokens += final.usage.output_tokens;
    webSearches += final.usage.server_tool_use?.web_search_requests ?? 0;

    if (final.stop_reason !== "pause_turn") break;
    // Il server ha messo in pausa un turno lungo di ricerca: rimandiamo il
    // contenuto parziale e il server riprende da dove si era fermato.
    messages.push({ role: "assistant", content: final.content });
  }

  if (final?.stop_reason === "refusal") {
    const msg = "Mi dispiace, non posso aiutarti con questa richiesta.";
    if (!answer) {
      answer = msg;
      onEvent({ type: "text", text: msg });
    }
  }

  return {
    answer,
    sources: [...sources.values()],
    stopReason: final?.stop_reason ?? null,
    model: final?.model ?? config.model,
    usage: { inputTokens, outputTokens, webSearches },
  };
}
