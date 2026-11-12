// Configurazione centrale dell'assistente, letta dalle variabili d'ambiente.

const EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"] as const;
export type Effort = (typeof EFFORT_LEVELS)[number];

function intFromEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function effortFromEnv(): Effort {
  const raw = process.env.EFFORT as Effort | undefined;
  return raw && EFFORT_LEVELS.includes(raw) ? raw : "medium";
}

export const config = {
  /** Nome provvisorio: cambialo con ASSISTANT_NAME quando il nome è deciso. */
  assistantName: process.env.ASSISTANT_NAME || "Ticly AI",
  model: "claude-opus-5-5",
  effort: effortFromEnv(),
  port: intFromEnv("PORT", 3000),
  /** Se impostata, i client devono mandare "Authorization: Bearer <chiave>". */
  apiKey: process.env.TICLY_API_KEY || "",
  maxSearches: intFromEnv("MAX_SEARCHES", 5),
  maxFetches: intFromEnv("MAX_FETCHES", 5),
  /** Quante volte riprendere un turno messo in pausa dal server (pause_turn). */
  maxContinuations: 5,
  /** Limiti sulla richiesta in ingresso. */
  maxMessages: 100,
  maxMessageChars: 50_000,
};
