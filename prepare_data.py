#!/usr/bin/env python3
"""Passo 2: trasforma i testi in token pronti per l'addestramento.

Testo libero (pre-addestramento: il modello impara la lingua):
    python prepare_data.py --text dati/*.txt --out out/pretrain

Conversazioni (fine-tuning: il modello impara a fare l'assistente):
    python prepare_data.py --chat dati/chat.jsonl --out out/chat

Ogni riga del file .jsonl è una conversazione:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
Per insegnare a usare la ricerca web, metti prima della domanda un messaggio
con role "context" che contiene i risultati (vedi data/esempio/chat.jsonl).
"""

import argparse
import json
import os
import random

import numpy as np

from ticly_ai.chat_format import encode_conversation
from ticly_ai.tokenizer import Tokenizer


def text_documents(paths, tok):
    """Un documento per file; i file grandi vengono letti a blocchi di righe."""
    bos = tok.token_id("<|bos|>")
    for path in paths:
        yield [bos]
        with open(path, encoding="utf-8", errors="replace") as f:
            while True:
                lines = f.readlines(4_000_000)
                if not lines:
                    break
                yield tok.encode("".join(lines))


def chat_documents(paths, tok):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    yield encode_conversation(tok, json.loads(line)["messages"])
                except (KeyError, ValueError, TypeError) as e:
                    raise SystemExit(f"{path}, riga {n}: conversazione non valida ({e})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", nargs="+", default=[], help="file di testo libero")
    p.add_argument("--chat", nargs="+", default=[], help="file .jsonl di conversazioni")
    p.add_argument("--tokenizer", default="out/tokenizer.json")
    p.add_argument("--out", required=True, help="cartella di destinazione")
    p.add_argument("--val-fraction", type=float, default=0.02, help="parte dei dati tenuta per la verifica")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not args.text and not args.chat:
        p.error("serve --text oppure --chat")

    tok = Tokenizer.load(args.tokenizer)
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    counts = {"train": 0, "val": 0}
    files = {name: open(os.path.join(args.out, f"{name}.bin"), "wb") for name in counts}

    def write(name, ids):
        np.asarray(ids, dtype=np.uint16).tofile(files[name])
        counts[name] += len(ids)

    # Conversazioni: ognuna va per intero in train o in val.
    for ids in chat_documents(args.chat, tok):
        write("val" if rng.random() < args.val_fraction else "train", ids)
    # Testo libero: la coda di ogni blocco va in val.
    for ids in text_documents(args.text, tok):
        cut = len(ids) - int(len(ids) * args.val_fraction)
        write("train", ids[:cut])
        write("val", ids[cut:])

    for f in files.values():
        f.close()
    if counts["val"] == 0:
        raise SystemExit("Dati di verifica vuoti: aggiungi dati o aumenta --val-fraction")
    print(f"train: {counts['train']:,} token   val: {counts['val']:,} token   -> {args.out}/")


if __name__ == "__main__":
    main()
