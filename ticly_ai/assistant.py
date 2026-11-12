"""L'assistente: modello + tokenizer + ricerca web, usato da chat.py e server.py."""

from __future__ import annotations

import codecs
import threading

from . import search as web
from .chat_format import build_prompt
from .model import load_checkpoint, pick_device
from .tokenizer import Tokenizer


class Assistant:
    def __init__(self, model_path: str, tokenizer_path: str, device: str | None = None):
        self.device = device or pick_device()
        self.tok = Tokenizer.load(tokenizer_path)
        self.model, _ = load_checkpoint(model_path, self.device)
        self.model.eval()
        if self.model.cfg.vocab_size != self.tok.vocab_size:
            raise ValueError("Il tokenizer non corrisponde a quello usato per addestrare il modello")
        self._lock = threading.Lock()  # una generazione alla volta
        # In risposta il modello si ferma a <|end|>; gli altri token speciali
        # qui non dovrebbero comparire e chiudono comunque la risposta.
        self._stop = tuple(self.tok.special.values())

    def search(self, query: str, max_results: int = 3) -> list[web.Result]:
        """Cerca sul web. In caso di errore di rete restituisce una lista vuota."""
        # Spazio per le pagine: circa metà del contesto (stima ~3 caratteri per token).
        chars = max(300, self.model.cfg.context_size * 3 // 2 // max_results)
        try:
            return web.search_and_read(query, max_results, chars_per_page=chars)
        except Exception:
            return []

    def reply(
        self,
        messages: list[dict],
        sources: list[web.Result] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """Genera la risposta all'ultimo messaggio, un pezzo di testo alla volta.

        `messages` è la conversazione ({"role": "user"|"assistant", "content"}).
        `sources`, se presenti, vengono inseriti come contesto prima dell'ultima domanda.
        """
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        if sources:
            msgs.insert(len(msgs) - 1, {"role": "context", "content": web.format_context(sources)})

        ctx = self.model.cfg.context_size
        reserve = min(max_tokens, ctx // 2)
        prompt = build_prompt(self.tok, msgs, ctx, reserve=reserve)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        with self._lock:
            for token in self.model.generate(
                prompt, max_new_tokens=max_tokens, temperature=temperature, stop_ids=self._stop
            ):
                # Un carattere può essere spezzato su più token: il decoder
                # incrementale emette testo solo quando i byte sono completi.
                text = decoder.decode(self.tok.decode_bytes(token))
                if text:
                    yield text
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail
