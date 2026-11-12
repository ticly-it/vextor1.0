#!/usr/bin/env python3
"""Parla con il modello dal terminale.

    python chat.py --model out/chat-model/best.pt
    python chat.py --model out/chat-model/best.pt --no-web     # senza ricerca web
"""

import argparse

from milo.assistant import Assistant


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default="out/tokenizer.json")
    p.add_argument("--no-web", action="store_true", help="non cercare sul web")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()

    assistant = Assistant(args.model, args.tokenizer)
    print("Scrivi una domanda (Ctrl+C per uscire, /nuova per ricominciare).")
    history: list[dict] = []
    while True:
        try:
            question = input("\nTu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question == "/nuova":
            history = []
            continue

        history.append({"role": "user", "content": question})
        sources = [] if args.no_web else assistant.search(question)
        for i, s in enumerate(sources, 1):
            print(f"  [{i}] {s.title} - {s.url}")

        print("AI: ", end="", flush=True)
        answer = ""
        for piece in assistant.reply(history, sources, args.max_tokens, args.temperature):
            print(piece, end="", flush=True)
            answer += piece
        print()
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
