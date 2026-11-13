#!/usr/bin/env python3
"""Parla con il modello dal terminale.

    python chat.py --model out/chat-model/best.pt
    python chat.py --model out/chat-model/best.pt --no-web     # senza ricerca web
    python chat.py --open-model out/milo-aperto                # Milo basato su un modello aperto
"""

import argparse

from milo.assistant import make_assistant


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="modello addestrato da zero (file .pt)")
    p.add_argument("--open-model", help="cartella dell'adattatore creato da finetune_open.py")
    p.add_argument("--tokenizer", default="out/tokenizer.json")
    p.add_argument("--no-web", action="store_true", help="non cercare sul web")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()

    if bool(args.model) == bool(args.open_model):
        p.error("indica --model oppure --open-model")
    assistant = make_assistant(args.model, args.open_model, args.tokenizer)
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
