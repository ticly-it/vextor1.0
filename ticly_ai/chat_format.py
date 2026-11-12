"""Formato delle conversazioni, uguale in addestramento e in uso.

    <|bos|>
    <|context|>risultati della ricerca web (facoltativo)<|end|>
    <|user|>domanda<|end|>
    <|assistant|>risposta<|end|>
    ...

Il blocco <|context|> precede la domanda a cui si riferisce: è così che il
modello impara a rispondere usando le pagine trovate sul web.
"""

from __future__ import annotations

from .tokenizer import Tokenizer

ROLES = {"user": "<|user|>", "assistant": "<|assistant|>", "context": "<|context|>"}


def encode_message(tok: Tokenizer, role: str, content: str) -> list[int]:
    if role not in ROLES:
        raise ValueError(f"Ruolo sconosciuto: {role!r} (validi: {', '.join(ROLES)})")
    return [tok.token_id(ROLES[role]), *tok.encode(content.strip()), tok.token_id("<|end|>")]


def encode_conversation(tok: Tokenizer, messages: list[dict]) -> list[int]:
    """Conversazione completa -> ID (per i dati di addestramento)."""
    ids = [tok.token_id("<|bos|>")]
    for m in messages:
        ids += encode_message(tok, m["role"], m["content"])
    return ids


def build_prompt(tok: Tokenizer, messages: list[dict], context_size: int, reserve: int = 256) -> list[int]:
    """Prompt per generare la prossima risposta dell'assistente.

    Se la conversazione non entra nel contesto si scartano i messaggi più
    vecchi, lasciando `reserve` token liberi per la risposta.
    """
    budget = context_size - reserve - 2  # <|bos|> e <|assistant|>
    encoded = [encode_message(tok, m["role"], m["content"]) for m in messages]

    kept: list[list[int]] = []
    used = 0
    for ids in reversed(encoded):
        if used + len(ids) > budget:
            break
        kept.append(ids)
        used += len(ids)
    if not kept:
        # Anche l'ultimo messaggio da solo è troppo lungo: se ne tiene la fine.
        last = encoded[-1]
        kept = [[last[0], *last[1:-1][-(budget - 2) :], last[-1]]]

    ids = [tok.token_id("<|bos|>")]
    for part in reversed(kept):
        ids += part
    ids.append(tok.token_id("<|assistant|>"))
    return ids
