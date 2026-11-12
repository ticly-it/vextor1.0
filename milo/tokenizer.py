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

import heapq
import json
import re
from collections import Counter, defaultdict

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
        words = [list(c.encode("utf-8")) for c in chunk_counts]
        freqs = list(chunk_counts.values())

        # Conteggio di ogni coppia e, per ogni coppia, in quali pezzi compare:
        # a ogni fusione si aggiornano solo i pezzi toccati.
        pairs: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for w, ids in enumerate(words):
            for pair in zip(ids, ids[1:]):
                pairs[pair] += freqs[w]
                where[pair].add(w)
        heap = [(-n, pair) for pair, n in pairs.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        while len(merges) < num_merges and heap:
            neg, best = heapq.heappop(heap)
            if pairs.get(best, 0) != -neg:
                continue  # voce vecchia: il conteggio nel frattempo è cambiato
            if -neg < 2:
                break
            new_id = 256 + len(merges)
            merges.append(best)

            changed: set[tuple[int, int]] = set()
            for w in where.pop(best, ()):
                ids, n = words[w], freqs[w]
                for pair in zip(ids, ids[1:]):
                    pairs[pair] -= n
                    changed.add(pair)
                ids = words[w] = list(_merge(tuple(ids), best, new_id))
                for pair in zip(ids, ids[1:]):
                    pairs[pair] += n
                    where[pair].add(w)
                    changed.add(pair)
            pairs.pop(best, None)
            for pair in changed:
                n = pairs.get(pair, 0)
                if n > 0:
                    heapq.heappush(heap, (-n, pair))
                else:
                    pairs.pop(pair, None)
            if verbose and len(merges) % 1000 == 0:
                print(f"  fusione {len(merges)}/{num_merges}", flush=True)

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
