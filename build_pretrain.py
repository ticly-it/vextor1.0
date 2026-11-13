#!/usr/bin/env python3
"""Costruisce i dati di pre-addestramento: scarica testo italiano e lo trasforma in token.

    # 1) campione per imparare il vocabolario
    python build_pretrain.py sample --out dati/campione.txt --mb 150
    python train_tokenizer.py --input dati/campione.txt --vocab-size 32000 --out out/tokenizer.json
    # 2) tutto il dataset, già in token
    python build_pretrain.py tokens --tokenizer out/tokenizer.json --out out/pretrain --target-tokens 3e9

Fonti (tutte su Hugging Face, in italiano):
    wikipedia   wikimedia/wikipedia 20231101.it      enciclopedia, fino al 50% dei token
    wikisource  wikimedia/wikisource 20231201.it     testi classici e documenti (es. Costituzione)
    libri       manu/project_gutenberg, split "it"   libri di pubblico dominio
    web         epfml/FineWeb2-HQ ita_Latn           pagine web di qualità alta, riempie il resto

Il testo non viene salvato, solo i token (2 byte l'uno): 3 miliardi di token = 6 GB.
Ogni file sorgente diventa un pezzo (train-<fonte>-<n>.bin + val-...); i pezzi finiti sono
registrati in manifest.json, quindi se il processo si interrompe riparte da dove era.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from collections import deque
from multiprocessing import Pool

import numpy as np

HF = "datasets/"


def clean_gutenberg(text: str) -> str:
    """Toglie l'intestazione e la licenza in inglese di Project Gutenberg."""
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n", text)
    if start:
        text = text[start.end():]
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG|End of (the )?Project Gutenberg", text)
    if end:
        text = text[: end.start()]
    # crediti dei volontari ("Produced by ...") all'inizio
    text = re.sub(r"^\s*(Produced by|This file was produced|E-text prepared by)[\s\S]*?\n\s*\n", "", text, count=1)
    return text.strip()


def with_title(row: dict) -> str:
    return f"{row['title']}\n\n{row['text'].strip()}"


# nome, cartella su HF, filtro sui file, colonne, funzione testo, quota massima dei token
SOURCES = [
    ("wikisource", "wikimedia/wikisource/20231201.it", "", ["title", "text"], with_title, 0.05),
    ("libri", "manu/project_gutenberg/data", "it-", ["text"], lambda r: clean_gutenberg(r["text"]), 0.05),
    ("wikipedia", "wikimedia/wikipedia/20231101.it", "", ["title", "text"], with_title, 0.50),
    ("web", "epfml/FineWeb2-HQ/ita_Latn", "", ["text"], lambda r: r["text"].strip(), 1.00),
]
MIN_CHARS = 200


def list_files(fs, folder: str, prefix: str) -> list[str]:
    files = sorted(f for f in fs.glob(f"{HF}{folder}/*.parquet") if os.path.basename(f).startswith(prefix))
    if not files:
        raise SystemExit(f"Nessun file trovato in {folder}")
    return files


def spread(files: list[str], rng: random.Random) -> list[str]:
    """Ordine casuale ma riproducibile: con il web si pescano file di periodi diversi."""
    files = list(files)
    rng.shuffle(files)
    return files


def iter_docs(fs, path: str, columns: list[str], to_text):
    import pyarrow.parquet as pq

    for attempt in range(5):
        try:
            with fs.open(path, "rb", block_size=16 << 20) as f:
                for batch in pq.ParquetFile(f).iter_batches(columns=columns, batch_size=500):
                    for row in batch.to_pylist():
                        text = to_text(row)
                        if len(text) >= MIN_CHARS:
                            yield text
            return
        except Exception as e:  # rete: si riprova il file da capo
            if attempt == 4:
                raise
            print(f"  errore di rete ({e!r}), riprovo...", flush=True)
            time.sleep(10 * (attempt + 1))


# ------------------------------------------------------------------ campione
def cmd_sample(args):
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    rng = random.Random(args.seed)
    # quote del campione: stesse proporzioni circa del dataset finale
    shares = {"wikisource": 0.05, "libri": 0.05, "wikipedia": 0.45, "web": 0.45}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for name, folder, prefix, cols, to_text, _ in SOURCES:
            budget = int(args.mb * 1e6 * shares[name])
            written = 0
            for path in spread(list_files(fs, folder, prefix), rng):
                for text in iter_docs(fs, path, cols, to_text):
                    written += out.write(text[:20000] + "\n\n")
                    if written >= budget:
                        break
                if written >= budget:
                    break
            print(f"{name}: {written / 1e6:.0f} MB", flush=True)
    print(f"Campione salvato in {args.out}")


# -------------------------------------------------------------------- token
_tok = None


def _init_worker(tokenizer_path):
    global _tok
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from milo.tokenizer import Tokenizer

    _tok = Tokenizer.load(tokenizer_path)


def _encode_batch(job):
    is_val, texts = job
    bos = _tok.token_id("<|bos|>")
    out = []
    for t in texts:
        out.append(bos)
        out.extend(_tok.encode(t))
    return is_val, np.asarray(out, dtype=np.uint16)


def cmd_tokens(args):
    from huggingface_hub import HfFileSystem

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from milo.tokenizer import Tokenizer

    vocab = Tokenizer.load(args.tokenizer).vocab_size
    if vocab > 65536:
        raise SystemExit("Vocabolario troppo grande per token a 16 bit")

    fs = HfFileSystem()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    tmp_dir = args.tmp or os.path.join(args.out, ".in-corso")
    os.makedirs(tmp_dir, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    manifest = {"tokenizer_vocab": vocab, "shards": {}}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("tokenizer_vocab") != vocab:
            raise SystemExit("La cartella contiene token di un altro tokenizer: usa una cartella nuova")

    def save_manifest():
        with open(manifest_path + ".tmp", "w") as f:
            json.dump(manifest, f, indent=1)
        os.replace(manifest_path + ".tmp", manifest_path)

    target = int(args.target_tokens)
    if manifest.get("complete") == target:
        print(f"Dati già pronti in {args.out} ({target / 1e9:.2f} miliardi di token)")
        return
    total = lambda: sum(s["train"] + s["val"] for s in manifest["shards"].values())  # noqa: E731
    print(f"Obiettivo: {target / 1e9:.2f} miliardi di token, già pronti: {total() / 1e9:.2f}", flush=True)

    workers = args.workers or os.cpu_count() or 1
    every = max(2, round(1 / args.val_fraction))
    with Pool(workers, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
        for name, folder, prefix, cols, to_text, share in SOURCES:
            cap = target if name == "web" else int(target * share)
            files = spread(list_files(fs, folder, prefix), rng)
            if args.max_files:
                files = files[: args.max_files]
            src_tokens = lambda: sum(s["train"] + s["val"] for k, s in manifest["shards"].items() if s["source"] == name)  # noqa: E731

            for path in files:
                key = path[len(HF):]
                if key in manifest["shards"]:
                    continue
                left = min(cap - src_tokens(), target - total())
                if left <= 0:
                    break
                n = len([s for s in manifest["shards"].values() if s["source"] == name])
                base = f"{name}-{n:03d}"
                tmp = {k: os.path.join(tmp_dir, f"{k}-{base}.bin") for k in ("train", "val")}
                counts = {"train": 0, "val": 0}
                t0 = time.time()
                print(f"[{name}] {os.path.basename(path)} (mancano {left / 1e6:.0f}M token per questa fonte)", flush=True)

                # Lotti di ~1 MB di testo; al massimo 4 lotti per processo in lavorazione,
                # così la memoria resta bassa e la quota della fonte non viene superata.
                def batches():
                    buf, size = [], 0
                    for text in iter_docs(fs, path, cols, to_text):
                        buf.append(text)
                        size += len(text)
                        if size >= 1_000_000:
                            yield buf
                            buf, size = [], 0
                    if buf:
                        yield buf

                pending = deque()

                def collect(block=True):
                    while pending and (block or pending[0].ready()):
                        is_val, arr = pending.popleft().get()
                        (fv if is_val else ft).write(arr.tobytes())
                        counts["val" if is_val else "train"] += len(arr)

                with open(tmp["train"], "wb") as ft, open(tmp["val"], "wb") as fv:
                    for i, buf in enumerate(batches()):
                        # un lotto ogni `every` va nei dati di verifica (deterministico,
                        # così l'ordine dei file resta identico anche quando si riprende)
                        is_val = i % every == every // 2
                        pending.append(pool.apply_async(_encode_batch, ((is_val, buf),)))
                        while len(pending) >= 4 * workers:
                            pending[0].wait()
                            collect(block=False)
                        collect(block=False)
                        # stima prudente: ~3,5 caratteri per token
                        in_flight = sum(1 for _ in pending) * 1_000_000 // 3
                        if counts["train"] + counts["val"] + in_flight >= left:
                            break
                    collect()
                # pezzo finito: lo si sposta nella cartella finale e lo si registra
                for k in ("train", "val"):
                    shutil.move(tmp[k], os.path.join(args.out, f"{k}-{base}.bin"))
                manifest["shards"][key] = {"source": name, "file": base, **counts}
                save_manifest()
                done = counts["train"] + counts["val"]
                print(f"   {done / 1e6:.0f}M token in {time.time() - t0:.0f}s  |  totale {total() / 1e9:.2f} miliardi", flush=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    manifest["complete"] = target
    save_manifest()
    by_src = {}
    for s in manifest["shards"].values():
        by_src[s["source"]] = by_src.get(s["source"], 0) + s["train"] + s["val"]
    print("Fatto. Token per fonte: " + ", ".join(f"{k} {v / 1e6:.0f}M" for k, v in by_src.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="campione di testo per il tokenizer")
    s.add_argument("--out", required=True)
    s.add_argument("--mb", type=float, default=150)
    s.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("tokens", help="dataset completo in token")
    t.add_argument("--tokenizer", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--target-tokens", type=float, default=3e9)
    t.add_argument("--val-fraction", type=float, default=0.005)
    t.add_argument("--tmp", help="cartella locale per i pezzi in lavorazione (es. su Colab /content/tmp)")
    t.add_argument("--workers", type=int, default=0, help="processi per la tokenizzazione (default: tutti i core)")
    t.add_argument("--max-files", type=int, default=0, help="solo per prove: massimo file per fonte")
    t.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    cmd_sample(args) if args.cmd == "sample" else cmd_tokens(args)


if __name__ == "__main__":
    main()
