import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { timingSafeEqual } from "node:crypto";
import Anthropic from "@anthropic-ai/sdk";
import { config } from "./config.js";
import { runAssistant, type AssistantEvent, type ChatMessage } from "./assistant.js";

class HttpError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
}

function isAuthorized(req: IncomingMessage): boolean {
  if (!config.apiKey) return true;
  const header = req.headers.authorization ?? "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : "";
  const a = Buffer.from(token);
  const b = Buffer.from(config.apiKey);
  return a.length === b.length && timingSafeEqual(a, b);
}

async function readJson(req: IncomingMessage, limitBytes = 2_000_000): Promise<unknown> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    size += (chunk as Buffer).length;
    if (size > limitBytes) throw new HttpError(413, "Richiesta troppo grande");
    chunks.push(chunk as Buffer);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw new HttpError(400, "JSON non valido");
  }
}

/**
 * Accetta due forme:
 *   { "message": "domanda" }                       -> domanda singola
 *   { "messages": [{ "role": "user", "content": "..." }, ...] } -> conversazione
 */
function parseChatRequest(body: unknown): { messages: ChatMessage[]; stream: boolean } {
  if (typeof body !== "object" || body === null) throw new HttpError(400, "Body mancante");
  const b = body as Record<string, unknown>;
  const stream = b.stream === true;

  let messages: ChatMessage[];
  if (typeof b.message === "string") {
    messages = [{ role: "user", content: b.message }];
  } else if (Array.isArray(b.messages)) {
    messages = b.messages.map((m, i) => {
      const msg = m as Record<string, unknown>;
      if ((msg.role !== "user" && msg.role !== "assistant") || typeof msg.content !== "string") {
        throw new HttpError(400, `messages[${i}]: servono "role" (user|assistant) e "content" (stringa)`);
      }
      return { role: msg.role, content: msg.content };
    });
  } else {
    throw new HttpError(400, 'Serve "message" (stringa) oppure "messages" (array)');
  }

  if (messages.length === 0) throw new HttpError(400, "Nessun messaggio");
  if (messages.length > config.maxMessages) throw new HttpError(400, "Troppi messaggi");
  if (messages[0].role !== "user") throw new HttpError(400, "Il primo messaggio deve essere dell'utente");
  if (messages[messages.length - 1].role !== "user") {
    throw new HttpError(400, "L'ultimo messaggio deve essere dell'utente");
  }
  for (const m of messages) {
    if (!m.content.trim()) throw new HttpError(400, "Messaggio vuoto");
    if (m.content.length > config.maxMessageChars) throw new HttpError(400, "Messaggio troppo lungo");
  }
  return { messages, stream };
}

function upstreamError(err: unknown): { status: number; message: string } {
  if (err instanceof Anthropic.RateLimitError) return { status: 429, message: "Troppe richieste, riprova tra poco" };
  if (err instanceof Anthropic.AuthenticationError) return { status: 500, message: "Configurazione del server non valida" };
  if (err instanceof Anthropic.BadRequestError) return { status: 400, message: err.message };
  if (err instanceof Anthropic.APIError) return { status: 502, message: "Servizio AI non disponibile" };
  return { status: 500, message: "Errore interno" };
}

async function handleChat(req: IncomingMessage, res: ServerResponse) {
  const { messages, stream } = parseChatRequest(await readJson(req));

  // Interrompe la generazione se il client chiude la connessione.
  const abort = new AbortController();
  res.on("close", () => {
    if (!res.writableFinished) abort.abort();
  });

  if (!stream) {
    const result = await runAssistant(messages, undefined, abort.signal);
    sendJson(res, 200, result);
    return;
  }

  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });
  const send = (event: string, data: unknown) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  try {
    const result = await runAssistant(
      messages,
      (e: AssistantEvent) => {
        const { type, ...data } = e;
        send(type, data);
      },
      abort.signal,
    );
    send("done", result);
  } catch (err) {
    if (!abort.signal.aborted) {
      console.error(err);
      send("error", upstreamError(err));
    }
  } finally {
    res.end();
  }
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", "http://localhost");
  try {
    if (req.method === "GET" && url.pathname === "/health") {
      sendJson(res, 200, { status: "ok", name: config.assistantName, model: config.model });
      return;
    }
    if (url.pathname === "/v1/chat") {
      if (req.method !== "POST") throw new HttpError(405, "Usa POST");
      if (!isAuthorized(req)) throw new HttpError(401, "Non autorizzato");
      await handleChat(req, res);
      return;
    }
    throw new HttpError(404, "Endpoint non trovato");
  } catch (err) {
    if (res.headersSent) {
      res.end();
      return;
    }
    if (err instanceof HttpError) {
      sendJson(res, err.status, { error: err.message });
    } else {
      console.error(err);
      const { status, message } = upstreamError(err);
      sendJson(res, status, { error: message });
    }
  }
});

server.listen(config.port, () => {
  console.log(`${config.assistantName} in ascolto su http://localhost:${config.port}`);
  if (!config.apiKey) console.warn("Attenzione: TICLY_API_KEY non impostata, l'API è aperta a chiunque.");
});
