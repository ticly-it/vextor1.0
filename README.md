# Ticly AI (nome provvisorio)

Assistente AI generalista di Ticly, la "Gemini" di Ticly: risponde a domande di ogni tipo
e **cerca sul web** (ricerca + lettura di pagine) per dare risposte aggiornate con le fonti.

Vextor resta l'AI per il codice; questo progetto è l'assistente generale.

- Modello: Claude Opus 5.5 (Anthropic) con i tool lato server `web_search` e `web_fetch`
- Solo API HTTP (Node + TypeScript), con risposta normale o in streaming (SSE)
- Il nome si cambia con la variabile `ASSISTANT_NAME`, senza toccare il codice

## Avvio

```bash
npm install
cp .env.example .env   # poi inserisci ANTHROPIC_API_KEY e TICLY_API_KEY
npm run dev            # sviluppo (riavvio automatico)
# oppure
npm run build && npm start
```

## Variabili d'ambiente

| Variabile | Default | Descrizione |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Chiave API Anthropic (obbligatoria) |
| `ASSISTANT_NAME` | `Ticly AI` | Nome dell'assistente |
| `PORT` | `3000` | Porta HTTP |
| `TICLY_API_KEY` | vuota | Se impostata, i client devono inviare `Authorization: Bearer <chiave>` |
| `EFFORT` | `medium` | Profondità di ragionamento: `low`, `medium`, `high`, `xhigh`, `max` |
| `MAX_SEARCHES` | `5` | Ricerche web massime per risposta |
| `MAX_FETCHES` | `5` | Pagine web lette al massimo per risposta |

## API

### `GET /health`

```json
{ "status": "ok", "name": "Ticly AI", "model": "claude-opus-5-5" }
```

### `POST /v1/chat`

L'API è senza stato: per una conversazione il client rimanda tutta la cronologia.

Domanda singola:

```bash
curl -X POST http://localhost:3000/v1/chat \
  -H "Authorization: Bearer $TICLY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "Quali sono le ultime novità sull'\''intelligenza artificiale?"}'
```

Conversazione:

```json
{
  "messages": [
    { "role": "user", "content": "Chi ha vinto l'ultimo Giro d'Italia?" },
    { "role": "assistant", "content": "..." },
    { "role": "user", "content": "E l'anno prima?" }
  ]
}
```

Risposta:

```json
{
  "answer": "Testo della risposta in Markdown...",
  "sources": [{ "url": "https://...", "title": "..." }],
  "stopReason": "end_turn",
  "model": "claude-opus-5-5",
  "usage": { "inputTokens": 1234, "outputTokens": 456, "webSearches": 2 }
}
```

### Streaming

Aggiungi `"stream": true` al body. La risposta è `text/event-stream` con questi eventi:

| Evento | Dati |
|---|---|
| `search` | `{ "query": "..." }` — l'assistente sta cercando sul web |
| `fetch` | `{ "url": "..." }` — l'assistente sta leggendo una pagina |
| `source` | `{ "source": { "url", "title" } }` — nuova fonte trovata |
| `text` | `{ "text": "..." }` — pezzo di risposta |
| `done` | stesso oggetto della risposta non in streaming |
| `error` | `{ "status", "message" }` |

```bash
curl -N -X POST http://localhost:3000/v1/chat \
  -H "Authorization: Bearer $TICLY_API_KEY" \
  -d '{"message": "Che tempo fa domani a Roma?", "stream": true}'
```

## Struttura

- `src/config.ts` — configurazione (nome, modello, limiti)
- `src/assistant.ts` — logica dell'assistente: prompt di sistema, ricerca web, fonti
- `src/server.ts` — server HTTP, validazione, autenticazione, streaming SSE
