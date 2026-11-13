"""Milo basato su un modello aperto già addestrato (es. Gemma 3) + adattatore LoRA.

Il modello di base resta quello originale; l'addestramento (finetune_open.py) produce
una cartella "adattatore" con i pochi pesi modificati e un file milo.json che dice
quale base usare. Qui c'è il formato delle conversazioni e la classe per usarlo.
"""

from __future__ import annotations

import json
import os
import threading

from . import search as web

SYSTEM_PROMPT = (
    "Sei Milo 1, l'assistente AI di Ticly. Rispondi in modo chiaro, accurato e diretto, "
    "nella lingua in cui ti scrive l'utente (italiano o inglese). "
    "Quando il messaggio contiene risultati di ricerca web numerati, basa la risposta su quelli "
    "e cita le fonti con il loro numero, ad esempio [1]; se non contengono la risposta, dillo. "
    "Non inventare fatti."
)
WEB_HEADER = "Risultati della ricerca web / Web search results:"


def to_chat(messages: list[dict], system: str = SYSTEM_PROMPT) -> list[dict]:
    """Dal formato di Milo (ruoli context/user/assistant) al formato standard dei modelli chat.

    Il blocco "context" viene messo dentro il messaggio dell'utente che lo segue.
    """
    out = [{"role": "system", "content": system}] if system else []
    pending = None
    for m in messages:
        role, content = m["role"], m["content"].strip()
        if role == "context":
            pending = content
        elif role == "user":
            if pending:
                content = f"{WEB_HEADER}\n{pending}\n\n{content}"
                pending = None
            out.append({"role": "user", "content": content})
        elif role == "assistant":
            out.append({"role": "assistant", "content": content})
        else:
            raise ValueError(f"Ruolo sconosciuto: {role!r}")
    return out


def load_base(base: str, four_bit: bool | None = None):
    """Carica tokenizer e modello di base. Su GPU usa 4 bit (serve bitsandbytes) per stare in memoria."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cuda = torch.cuda.is_available()
    if four_bit is None:
        four_bit = cuda
    kwargs = {}
    if cuda:
        bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        dtype = torch.bfloat16 if bf16 else torch.float16
        kwargs["device_map"] = {"": 0}
        if four_bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            )
    else:
        dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, **kwargs)
    model.generation_config.max_length = None  # la lunghezza la decide max_new_tokens
    return tok, model


class OpenAssistant:
    """Stessa interfaccia di milo.assistant.Assistant (search, reply)."""

    def __init__(self, adapter_dir: str, four_bit: bool | None = None):
        import torch
        from peft import PeftModel

        with open(os.path.join(adapter_dir, "milo.json"), encoding="utf-8") as f:
            self.info = json.load(f)
        self.system = self.info.get("system", SYSTEM_PROMPT)
        self.tok, base = load_base(self.info["base"], four_bit)
        self.model = PeftModel.from_pretrained(base, adapter_dir).eval()
        self.device = str(next(self.model.parameters()).device)
        self._torch = torch
        self._lock = threading.Lock()

    def search(self, query: str, max_results: int = 4) -> list[web.Result]:
        try:
            return web.search_and_read(query, max_results, chars_per_page=6000)
        except Exception:
            return []

    def reply(self, messages, sources=None, max_tokens: int = 1024, temperature: float = 0.7):
        from transformers import TextIteratorStreamer

        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        if sources:
            msgs.insert(len(msgs) - 1, {"role": "context", "content": web.format_context(sources)})
        inputs = self.tok.apply_chat_template(
            to_chat(msgs, self.system), add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        sampling = {"do_sample": True, "temperature": temperature, "top_p": 0.95} if temperature > 0 else {"do_sample": False}

        def run():
            with self._torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=max_tokens, streamer=streamer, **sampling)

        with self._lock:
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            for piece in streamer:
                if piece:
                    yield piece
            thread.join()
