#!/usr/bin/env python3
"""Scarica e prepara le conversazioni per il fine-tuning di Milo 1.

    python build_chat.py --out dati/chat.jsonl

Fonti (Hugging Face):
    FreedomIntelligence/alpaca-gpt4-italian   istruzioni con risposte generate da GPT-4 (~50k)
    teelinsan/camoscio                         istruzioni tradotte da Alpaca (~50k)
    OpenAssistant/oasst2                       conversazioni scritte da persone (solo italiano)
    crux82/squad_it                            domande su un testo: insegnano a usare le
                                               pagine trovate dalla ricerca web
    + conversazioni sull'identità di Milo 1, scritte qui sotto

Licenze: camoscio openrail, oasst2 apache-2.0, squad_it non dichiarata, alpaca-gpt4
deriva da risposte di GPT-4 (termini di OpenAI). Verificale prima di un uso commerciale.
"""

import argparse
import json
import os
import random
import re

NAME = "Milo 1"

# Conversazioni sull'identità: senza questi esempi il modello non sa chi è.
IDENTITY = [
    (["Chi sei?", "Ciao, chi sei?", "Con chi sto parlando?", "Presentati."],
     [f"Sono {NAME}, l'assistente AI di Ticly. Posso rispondere alle tue domande e cercare informazioni sul web.",
      f"Mi chiamo {NAME} e sono l'assistente AI di Ticly: rispondo alle domande e cerco informazioni sul web."]),
    (["Come ti chiami?", "Qual è il tuo nome?", "Hai un nome?"],
     [f"Mi chiamo {NAME}.", f"Il mio nome è {NAME}, l'assistente AI di Ticly."]),
    (["Chi ti ha creato?", "Chi ti ha sviluppato?", "Di chi sei?", "Chi ti ha fatto?"],
     [f"Sono stato creato da Ticly.", f"Mi ha sviluppato Ticly: sono {NAME}, un modello addestrato da zero."]),
    (["Sei ChatGPT?", "Sei Gemini?", "Sei Claude?", "Sei un modello di Google?", "Sei fatto da OpenAI?"],
     [f"No, sono {NAME}, l'assistente AI di Ticly.", f"No: mi chiamo {NAME} e sono stato creato da Ticly."]),
    (["Cosa sai fare?", "In cosa puoi aiutarmi?", "A cosa servi?", "Come puoi aiutarmi?"],
     ["Posso rispondere a domande, spiegare argomenti, aiutarti a scrivere e cercare informazioni aggiornate sul web.",
      "Rispondo alle domande, spiego concetti e cerco informazioni sul web quando servono dati aggiornati."]),
    (["Sei una persona?", "Sei umano?", "Sei un robot?"],
     [f"No, sono un'intelligenza artificiale: mi chiamo {NAME}.", f"Sono {NAME}, un assistente AI, non una persona."]),
    (["Ciao!", "Ciao", "Buongiorno", "Buonasera", "Salve", "Ehi"],
     ["Ciao! Come posso aiutarti?", f"Ciao! Sono {NAME}, dimmi pure.", "Ciao! Di cosa hai bisogno?"]),
    (["Grazie!", "Grazie mille", "Ti ringrazio", "Perfetto, grazie"],
     ["Prego! Se ti serve altro, sono qui.", "Di niente!", "Figurati, è stato un piacere."]),
]

NO_ANSWER = [
    "Nelle pagine che ho trovato non c'è questa informazione, quindi non posso risponderti con certezza.",
    "Le fonti trovate non contengono la risposta a questa domanda.",
]


def conversation(*turns):
    return {"messages": [{"role": r, "content": c} for r, c in turns]}


def identity_conversations(rng, copies):
    out = []
    for _ in range(copies):
        for questions, answers in IDENTITY:
            for q in questions:
                out.append(conversation(("user", q), ("assistant", rng.choice(answers))))
    return out


def from_camoscio(row):
    """Riga {instruction, input, output} -> conversazione, oppure None se inutilizzabile."""
    instruction = (row.get("instruction") or "").strip()
    extra = (row.get("input") or "").strip()
    output = (row.get("output") or "").strip()
    if not instruction or not output or len(output) < 2:
        return None
    question = f"{instruction}\n\n{extra}" if extra else instruction
    return conversation(("user", question), ("assistant", output))


def _page(title: str, text: str) -> tuple[str, str]:
    """Titolo e indirizzo finti ma plausibili per una pagina (squad_it non ha i titoli)."""
    title = (title or "").replace("_", " ").strip() or " ".join(text.split()[:6]).rstrip(",.;:")
    slug = re.sub(r"[^\w]+", "_", title).strip("_")[:60]
    return title, f"https://it.wikipedia.org/wiki/{slug}"


def from_squad(row, rng, distractors=(), other_question=None):
    """Riga {context, question, answers} -> conversazione con blocco "context".

    Il contesto imita i risultati di milo/search.py: alcune pagine numerate
    [1] titolo \n url \n testo, di cui una sola contiene la risposta. Il modello
    impara a trovare quella giusta e a citarla con il suo numero.
    Con `other_question` nessuna pagina contiene la risposta.
    """
    context = (row.get("context") or "").strip()
    question = (row.get("question") or "").strip()
    answers = (row.get("answers") or {}).get("text") or []
    if not context or not question or not answers:
        return None
    pages = [(row.get("title"), context, True)] + [(t, c, False) for t, c in distractors]
    rng.shuffle(pages)
    blocks, good = [], 0
    for i, (title, text, is_good) in enumerate(pages, 1):
        title, url = _page(title, text)
        blocks.append(f"[{i}] {title}\n{url}\n{text}")
        if is_good:
            good = i
    block = "\n\n".join(blocks)
    if other_question:
        return conversation(("context", block), ("user", other_question), ("assistant", rng.choice(NO_ANSWER)))
    answer = answers[0].strip().rstrip(".")
    if not answer:
        return None
    answer = answer[0].upper() + answer[1:]
    return conversation(("context", block), ("user", question), ("assistant", f"{answer} [{good}]."))


def from_alpaca_gpt4(row):
    turns = []
    for m in row.get("conversations") or []:
        role = {"human": "user", "gpt": "assistant"}.get(m.get("from"))
        text = (m.get("value") or "").strip()
        if not role or not text:
            return None
        turns.append((role, text))
    if len(turns) < 2 or turns[0][0] != "user" or turns[-1][0] != "assistant":
        return None
    return conversation(*turns)


def oasst_conversations(rows):
    """Alberi di messaggi di oasst2 -> conversazioni lineari in italiano.

    Da ogni domanda iniziale si segue, a ogni passo, la risposta migliore
    secondo le valutazioni (rank più basso).
    """
    by_id, children = {}, {}
    for r in rows:
        if r.get("lang") != "it" or r.get("deleted") or r.get("review_result") is False:
            continue
        by_id[r["message_id"]] = r
        children.setdefault(r.get("parent_id"), []).append(r)
    for kids in children.values():
        kids.sort(key=lambda r: (r.get("rank") is None, r.get("rank") or 0))
    out = []
    for root in children.get(None, []):
        turns, node = [], root
        while node:
            role = "user" if node["role"] == "prompter" else "assistant"
            turns.append((role, node["text"].strip()))
            kids = children.get(node["message_id"], [])
            node = kids[0] if kids else None
        if turns and turns[-1][0] == "user":
            turns.pop()
        if len(turns) >= 2:
            out.append(conversation(*turns))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="dati/chat.jsonl")
    p.add_argument("--max-chat", type=int, default=50000, help="massimo di conversazioni da ogni fonte")
    p.add_argument("--identity-copies", type=int, default=40, help="ripetizioni delle conversazioni sull'identità")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    failed = []

    # ------------------------------------------------------- conversazioni
    convs = identity_conversations(rng, args.identity_copies)
    print(f"identità: {len(convs)} conversazioni")

    def add(name, items):
        n = 0
        for c in items:
            if c:
                convs.append(c)
                n += 1
                if n >= args.max_chat:
                    break
        print(f"{name}: {n} conversazioni", flush=True)

    try:
        ds = load_dataset("FreedomIntelligence/alpaca-gpt4-italian", split="train").shuffle(seed=args.seed)
        add("alpaca-gpt4-italian", (from_alpaca_gpt4(r) for r in ds))
    except Exception as e:
        failed.append(f"alpaca-gpt4-italian ({e!r})")

    try:
        ds = load_dataset("teelinsan/camoscio", split="train").shuffle(seed=args.seed)
        add("camoscio", (from_camoscio(r) for r in ds))
    except Exception as e:
        failed.append(f"camoscio ({e!r})")

    try:
        rows = []
        for split in ("train", "validation"):
            rows += load_dataset("OpenAssistant/oasst2", split=split).select_columns(
                ["message_id", "parent_id", "text", "role", "lang", "rank", "deleted", "review_result"]
            ).to_list()
        add("oasst2 (italiano)", oasst_conversations(rows))
    except Exception as e:
        failed.append(f"oasst2 ({e!r})")

    try:
        ds = load_dataset("crux82/squad_it", split="train").shuffle(seed=args.seed)
        contexts = list({r["context"] for r in ds.select(range(min(5000, len(ds))))})

        def squad():
            previous = None
            for row in ds:
                k = rng.choice([0, 1, 2, 2])  # quante pagine "sbagliate" aggiungere
                distractors = [(None, c) for c in rng.sample(contexts, k) if c != row["context"]]
                # 1 esempio su 10: nessuna pagina contiene la risposta
                other = previous if previous and rng.random() < 0.1 else None
                yield from_squad(row, rng, distractors, other_question=other)
                previous = row["question"]

        add("squad_it (risposte dalle pagine web)", squad())
    except Exception as e:
        failed.append(f"squad_it ({e!r})")

    rng.shuffle(convs)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{args.out}: {len(convs)} conversazioni in totale")

    if failed:
        raise SystemExit("ATTENZIONE, fonti non scaricate:\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()
