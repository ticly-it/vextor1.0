#!/usr/bin/env python3
"""API HTTP dell'assistente.

    python server.py --model out/chat-model/best.pt --port 3000

    POST /v1/chat   {"message": "..."}  oppure  {"messages": [{"role": "user", "content": "..."}]}
                    opzionali: "web" (default true), "stream", "temperature", "max_tokens"
    GET  /health

Variabili d'ambiente:
    ASSISTANT_NAME   nome dell'assistente (default "Milo 1")
    TICLY_API_KEY    se impostata, serve l'header "Authorization: Bearer <chiave>"
"""

import argparse
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from milo.assistant import Assistant

NAME = os.environ.get("ASSISTANT_NAME", "Milo 1")
API_KEY = os.environ.get("TICLY_API_KEY", "")
MAX_BODY = 2_000_000
MAX_MESSAGES = 100
assistant: Assistant


class BadRequest(Exception):
    pass


def parse_request(body: dict) -> dict:
    if not isinstance(body, dict):
        raise BadRequest("Il body deve essere un oggetto JSON")
    if isinstance(body.get("message"), str):
        messages = [{"role": "user", "content": body["message"]}]
    elif isinstance(body.get("messages"), list):
        messages = body["messages"]
    else:
        raise BadRequest('Serve "message" (stringa) oppure "messages" (lista)')

    if not messages or len(messages) > MAX_MESSAGES:
        raise BadRequest(f"Servono da 1 a {MAX_MESSAGES} messaggi")
    for i, m in enumerate(messages):
        ok = isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
        if not ok or not m["content"].strip():
            raise BadRequest(f'messages[{i}]: servono "role" (user|assistant) e "content" non vuoto')
    if messages[-1]["role"] != "user":
        raise BadRequest("L'ultimo messaggio deve essere dell'utente")

    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 512)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise BadRequest('"temperature" deve essere tra 0 e 2')
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 4096:
        raise BadRequest('"max_tokens" deve essere tra 1 e 4096')
    return {
        "messages": messages,
        "web": body.get("web", True) is not False,
        "stream": body.get("stream") is True,
        "temperature": float(temperature),
        "max_tokens": max_tokens,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}")

    def send_json(self, status: int, data) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ok", "name": NAME})
        else:
            self.send_json(404, {"error": "Endpoint non trovato"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self.close_connection = True
            return self.send_json(413, {"error": "Richiesta troppo grande"})
        raw = self.rfile.read(length)

        if self.path != "/v1/chat":
            return self.send_json(404, {"error": "Endpoint non trovato"})
        token = (self.headers.get("Authorization") or "").removeprefix("Bearer ")
        if API_KEY and not hmac.compare_digest(token.encode(), API_KEY.encode()):
            return self.send_json(401, {"error": "Non autorizzato"})
        try:
            req = parse_request(json.loads(raw or b"{}"))
        except json.JSONDecodeError:
            return self.send_json(400, {"error": "JSON non valido"})
        except BadRequest as e:
            return self.send_json(400, {"error": str(e)})

        sources = assistant.search(req["messages"][-1]["content"]) if req["web"] else []
        source_list = [{"title": s.title, "url": s.url} for s in sources]
        pieces = assistant.reply(req["messages"], sources, req["max_tokens"], req["temperature"])

        if not req["stream"]:
            try:
                answer = "".join(pieces)
            except Exception as e:  # es. errore del modello
                print(f"Errore in generazione: {e!r}")
                return self.send_json(500, {"error": "Errore interno"})
            return self.send_json(200, {"answer": answer, "sources": source_list})

        # Streaming (Server-Sent Events): la connessione si chiude a fine risposta.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def event(name, data):
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        try:
            event("sources", {"sources": source_list})
            answer = ""
            for piece in pieces:
                answer += piece
                event("text", {"text": piece})
            event("done", {"answer": answer, "sources": source_list})
        except (BrokenPipeError, ConnectionResetError):
            pieces.close()  # il client ha chiuso: ferma la generazione
        except Exception as e:
            print(f"Errore in generazione: {e!r}")
            event("error", {"error": "Errore interno"})


def main():
    global assistant
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default="out/tokenizer.json")
    p.add_argument("--host", default="127.0.0.1", help="usa 0.0.0.0 per esporlo in rete")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 3000)))
    args = p.parse_args()

    assistant = Assistant(args.model, args.tokenizer)
    if not API_KEY:
        print("Attenzione: TICLY_API_KEY non impostata, l'API è aperta a chiunque la raggiunga.")
    print(f"{NAME} in ascolto su http://{args.host}:{args.port} (device: {assistant.device})")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
