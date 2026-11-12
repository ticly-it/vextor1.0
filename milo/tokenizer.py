"""Tokenizer BPE a livello di byte, scritto da zero.

Il testo viene spezzato in pezzi (parole, spazi, punteggiatura), ogni pezzo è una
sequenza di byte, e l'addestramento fonde ripetutamente la coppia di token più
frequente fino a raggiungere la dimensione del vocabolario voluta.

ID dei token:
    0..255            byte grezzi
    256..N-1          fusioni imparate
    N..N+len(SPECIAL) token speciali (ruoli della chat, ecc.)
"""

from __future__ import annotations

import json
import re
from collections import Counter

# Token speciali: non vengono mai prodotti dal testo normale, solo dal formato chat.
SPECIAL_TOKENS = ["<|bos|>", "<|end|>", "<|user|>", "<|assistant|>", "<|context|>"]

# Parole (con eventuale spazio davanti), numeri, punteggiatura, spazi.
_CHUNK_RE = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+|_+")
_SPECIAL_RE = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")


def _merge(ids: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: list[tuple[int, int]] = merges or []
        self._build()

    # ------------------------------------------------------------------ setup
    def _build(self) -> None:
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        base = len(self.vocab)
        self.special = {tok: base + i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.special_by_id = {i: tok for tok, i in self.special.items()}
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(SPECIAL_TOKENS)

    def token_id(self, special: str) -> int:
        return self.special[special]

    # --------------------------------------------------------------- training
    @classmethod
    def train(cls, texts, vocab_size: int, verbose: bool = True) -> "Tokenizer":
        """Impara le fusioni BPE da un iterabile di stringhe."""
        num_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
        if num_merges < 0:
            raise ValueError("vocab_size troppo piccolo")

        chunk_counts: Counter[str] = Counter()
        for text in texts:
            for part in _SPECIAL_RE.split(text):
                if part and part not in SPECIAL_TOKENS:
                    chunk_counts.update(_CHUNK_RE.findall(part))

        # Ogni pezzo distinto viene tenuto una volta sola, con la sua frequenza.
        words = [(tuple(c.encode("utf-8")), n) for c, n in chunk_counts.items()]
        merges: list[tuple[int, int]] = []

        for step in range(num_merges):
            pairs: Counter[tuple[int, int]] = Counter()
            for ids, n in words:
                for pair in zip(ids, ids[1:]):
                    pairs[pair] += n
            if not pairs:
                break
            best = max(pairs, key=lambda p: (pairs[p], p))
            if pairs[best] < 2:
                break
            new_id = 256 + len(merges)
            merges.append(best)
            a, b = best
            words = [
                (_merge(ids, best, new_id), n) if a in ids and b in ids else (ids, n)
                for ids, n in words
            ]
            if verbose and (step + 1) % 500 == 0:
                print(f"  fusione {step + 1}/{num_merges}")

        return cls(merges)

    # ---------------------------------------------------------- encode/decode
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = tuple(chunk.encode("utf-8"))
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            rank = self.ranks.get(pair)
            if rank is None:
                break
            ids = _merge(ids, pair, 256 + rank)
        out = list(ids)
        if len(self._cache) < 500_000:
            self._cache[chunk] = out
        return out

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Testo -> lista di ID.

        Con allow_special=False (default, per il testo scritto dagli utenti) le
        stringhe tipo "<|user|>" restano testo normale: così nessuno può
        iniettare ruoli finti nella conversazione.
        """
        out: list[int] = []
        parts = _SPECIAL_RE.split(text) if allow_special else [text]
        for part in parts:
            if not part:
                continue
            if allow_special and part in self.special:
                out.append(self.special[part])
            else:
                for chunk in _CHUNK_RE.findall(part):
                    out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids, skip_special: bool = True) -> str:
        data = bytearray()
        for i in ids:
            i = int(i)
            if i < len(self.vocab):
                data += self.vocab[i]
            elif not skip_special and i in self.special_by_id:
                data += self.special_by_id[i].encode("utf-8")
        return data.decode("utf-8", errors="replace")

    def decode_bytes(self, token_id: int) -> bytes:
        """Byte di un singolo token (vuoto per i token speciali). Serve allo streaming."""
        return self.vocab[token_id] if token_id < len(self.vocab) else b""

    # ------------------------------------------------------------------- file
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "special": SPECIAL_TOKENS, "merges": self.merges}, f)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("special") != SPECIAL_TOKENS:
            raise ValueError("Il tokenizer è stato creato con token speciali diversi")
        return cls([tuple(m) for m in data["merges"]])
