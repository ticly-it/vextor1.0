#!/usr/bin/env python3
"""Passo 1: impara il vocabolario (tokenizer BPE) dai tuoi testi.

    python train_tokenizer.py --input dati/*.txt dati/chat.jsonl --vocab-size 16000
"""

import argparse
import json

from ticly_ai.tokenizer import Tokenizer


def read_texts(paths, max_chars):
    """Legge file .txt (testo libero) e .jsonl (conversazioni)."""
    total = 0
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if path.endswith(".jsonl"):
                    if not line.strip():
                        continue
                    line = "\n".join(m["content"] for m in json.loads(line)["messages"])
                yield line
                total += len(line)
                if total >= max_chars:
                    return


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", required=True, help="file .txt e/o .jsonl")
    p.add_argument("--vocab-size", type=int, default=16000, help="massimo 65536")
    p.add_argument("--out", default="out/tokenizer.json")
    p.add_argument("--max-chars", type=int, default=200_000_000, help="quanti caratteri leggere al massimo")
    args = p.parse_args()
    if args.vocab_size > 65536:
        p.error("vocab-size massimo 65536 (i token sono salvati a 16 bit)")

    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok = Tokenizer.train(read_texts(args.input, args.max_chars), args.vocab_size)
    tok.save(args.out)
    print(f"Tokenizer salvato in {args.out} ({tok.vocab_size} token)")


if __name__ == "__main__":
    main()
