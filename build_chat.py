#!/usr/bin/env python3
"""Scarica e prepara le conversazioni per il fine-tuning di Milo 1, in italiano e in inglese.

    python build_chat.py --out dati/chat.jsonl
    python build_chat.py --out dati/chat.jsonl --lang it              # solo italiano
    python build_chat.py --out dati/chat-aperto.jsonl --max-chat 1500 --identity-copies 8 \
        --base-model "Gemma 3" --base-maker Google                    # per finetune_open.py

Fonti (Hugging Face):
  Italiano
    FreedomIntelligence/alpaca-gpt4-italian   istruzioni con risposte generate da GPT-4
    teelinsan/camoscio                         istruzioni tradotte da Alpaca
    crux82/squad_it                            domande su un testo -> "rispondi usando le pagine web"
  Inglese
    vicgalle/alpaca-gpt4                       istruzioni con risposte generate da GPT-4
    rajpurkar/squad                            domande su un testo -> "rispondi usando le pagine web"
  Entrambe
    OpenAssistant/oasst2                       conversazioni scritte da persone
    + conversazioni sull'identità di Milo 1, scritte qui sotto

Licenze: camoscio openrail, oasst2 apache-2.0, squad cc-by-sa-4.0, squad_it non dichiarata,
alpaca-gpt4 (it/en) deriva da risposte di GPT-4 (cc-by-nc e termini di OpenAI).
Verificale prima di un uso commerciale.
"""

import argparse
import json
import os
import random
import re

NAME = "Milo 1"

# Conversazioni sull'identità: senza questi esempi il modello non sa chi è.
# {tech} viene sostituito con la frase sulla tecnologia (vedi --base-model): Milo non deve
# negare di essere basato su un altro modello, se lo è.
IDENTITY = {
    "it": [
        (["Chi sei?", "Ciao, chi sei?", "Con chi sto parlando?", "Presentati."],
         [f"Sono {NAME}, l'assistente AI di Ticly. Posso rispondere alle tue domande e cercare informazioni sul web.",
          f"Mi chiamo {NAME} e sono l'assistente AI di Ticly: rispondo alle domande e cerco informazioni sul web."]),
        (["Come ti chiami?", "Qual è il tuo nome?", "Hai un nome?"],
         [f"Mi chiamo {NAME}.", f"Il mio nome è {NAME}, l'assistente AI di Ticly."]),
        (["Chi ti ha creato?", "Chi ti ha sviluppato?", "Di chi sei?", "Chi ti ha fatto?"],
         [f"Sono {NAME}, l'assistente AI di Ticly. {{tech}}"]),
        (["Sei ChatGPT?", "Sei Gemini?", "Sei Claude?", "Sei un modello di Google?", "Sei fatto da OpenAI?",
          "Su che modello sei basato?", "Che tecnologia usi?"],
         [f"Sono {NAME}, l'assistente AI di Ticly. {{tech}}"]),
        (["Cosa sai fare?", "In cosa puoi aiutarmi?", "A cosa servi?", "Come puoi aiutarmi?"],
         ["Posso rispondere a domande, spiegare argomenti, aiutarti a scrivere e cercare informazioni aggiornate sul web.",
          "Rispondo alle domande, spiego concetti e cerco informazioni sul web quando servono dati aggiornati."]),
        (["Sei una persona?", "Sei umano?", "Sei un robot?"],
         [f"No, sono un'intelligenza artificiale: mi chiamo {NAME}.", f"Sono {NAME}, un assistente AI, non una persona."]),
        (["Parli inglese?", "Sai l'inglese?", "In che lingue parli?"],
         ["Sì, parlo italiano e inglese: ti rispondo nella lingua in cui mi scrivi."]),
        (["Ciao!", "Ciao", "Buongiorno", "Buonasera", "Salve", "Ehi"],
         ["Ciao! Come posso aiutarti?", f"Ciao! Sono {NAME}, dimmi pure.", "Ciao! Di cosa hai bisogno?"]),
        (["Grazie!", "Grazie mille", "Ti ringrazio", "Perfetto, grazie"],
         ["Prego! Se ti serve altro, sono qui.", "Di niente!", "Figurati, è stato un piacere."]),
    ],
    "en": [
        (["Who are you?", "Hi, who are you?", "Who am I talking to?", "Introduce yourself."],
         [f"I'm {NAME}, Ticly's AI assistant. I can answer your questions and look things up on the web.",
          f"My name is {NAME} and I'm Ticly's AI assistant: I answer questions and search the web."]),
        (["What's your name?", "What is your name?", "Do you have a name?"],
         [f"My name is {NAME}.", f"I'm {NAME}, Ticly's AI assistant."]),
        (["Who made you?", "Who created you?", "Who developed you?", "Who built you?"],
         [f"I'm {NAME}, Ticly's AI assistant. {{tech}}"]),
        (["Are you ChatGPT?", "Are you Gemini?", "Are you Claude?", "Are you a Google model?", "Are you made by OpenAI?",
          "What model are you based on?", "What technology do you use?"],
         [f"I'm {NAME}, Ticly's AI assistant. {{tech}}"]),
        (["What can you do?", "How can you help me?", "What are you for?"],
         ["I can answer questions, explain topics, help you write, and search the web for up-to-date information.",
          "I answer questions, explain concepts, and search the web when fresh information is needed."]),
        (["Are you a person?", "Are you human?", "Are you a robot?"],
         [f"No, I'm an artificial intelligence. My name is {NAME}.", f"I'm {NAME}, an AI assistant, not a person."]),
        (["Do you speak Italian?", "What languages do you speak?"],
         ["Yes, I speak Italian and English: I reply in the language you write in."]),
        (["Hi!", "Hello", "Hey", "Good morning", "Good evening"],
         ["Hi! How can I help you?", f"Hello! I'm {NAME}. What do you need?", "Hi! What can I do for you?"]),
        (["Thanks!", "Thank you so much", "Thank you", "Great, thanks"],
         ["You're welcome! I'm here if you need anything else.", "No problem!", "Happy to help."]),
    ],
}

NO_ANSWER = {
    "it": ["Nelle pagine che ho trovato non c'è questa informazione, quindi non posso risponderti con certezza.",
           "Le fonti trovate non contengono la risposta a questa domanda."],
    "en": ["The pages I found don't contain this information, so I can't answer with certainty.",
           "The sources I found don't include the answer to this question."],
}
WIKI = {"it": "https://it.wikipedia.org/wiki/", "en": "https://en.wikipedia.org/wiki/"}


def tech_sentence(lang: str, base_model: str | None, base_maker: str | None) -> str:
    if base_model:
        by = {"it": f" di {base_maker}", "en": f" by {base_maker}"}[lang] if base_maker else ""
        return {"it": f"Sono basato su {base_model}, un modello aperto{by}, adattato da Ticly.",
                "en": f"I'm based on {base_model}, an open model{by}, adapted by Ticly."}[lang]
    return {"it": "Sono un modello costruito e addestrato da zero da Ticly.",
            "en": "I'm a model built and trained from scratch by Ticly."}[lang]


def conversation(*turns):
    return {"messages": [{"role": r, "content": c} for r, c in turns]}


def identity_conversations(rng, copies, langs, base_model=None, base_maker=None):
    out = []
    for lang in langs:
        tech = tech_sentence(lang, base_model, base_maker)
        for _ in range(copies):
            for questions, answers in IDENTITY[lang]:
                for q in questions:
                    out.append(conversation(("user", q), ("assistant", rng.choice(answers).format(tech=tech))))
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


def _page(title: str, text: str, lang: str) -> tuple[str, str]:
    """Titolo e indirizzo plausibili per una pagina (squad_it non ha i titoli)."""
    title = (title or "").replace("_", " ").strip() or " ".join(text.split()[:6]).rstrip(",.;:")
    slug = re.sub(r"[^\w]+", "_", title).strip("_")[:60]
    return title, WIKI[lang] + slug


def from_squad(row, rng, lang="it", distractors=(), other_question=None):
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
        title, url = _page(title, text, lang)
        blocks.append(f"[{i}] {title}\n{url}\n{text}")
        if is_good:
            good = i
    block = "\n\n".join(blocks)
    if other_question:
        return conversation(("context", block), ("user", other_question), ("assistant", rng.choice(NO_ANSWER[lang])))
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


def oasst_conversations(rows, lang):
    """Alberi di messaggi di oasst2 -> conversazioni lineari in una lingua.

    Da ogni domanda iniziale si segue, a ogni passo, la risposta migliore
    secondo le valutazioni (rank più basso).
    """
    by_id, children = {}, {}
    for r in rows:
        if r.get("lang") != lang or r.get("deleted") or r.get("review_result") is False:
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
    p.add_argument("--lang", nargs="+", choices=["it", "en"], default=["it", "en"])
    p.add_argument("--max-chat", type=int, default=50000, help="massimo di conversazioni da ogni fonte")
    p.add_argument("--identity-copies", type=int, default=40, help="ripetizioni delle conversazioni sull'identità")
    p.add_argument("--base-model", help='se Milo è basato su un modello aperto, il suo nome (es. "Gemma 3"): '
                                        "Milo lo dirà quando gli chiedono su cosa è basato")
    p.add_argument("--base-maker", help='chi ha creato il modello di base (es. "Google")')
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    failed = []

    convs = identity_conversations(rng, args.identity_copies, args.lang, args.base_model, args.base_maker)
    print(f"identità: {len(convs)} conversazioni", flush=True)

    def add(name, make):
        """Aggiunge fino a --max-chat conversazioni da una fonte; un errore non ferma le altre."""
        try:
            n = 0
            for c in make():
                if c:
                    convs.append(c)
                    n += 1
                    if n >= args.max_chat:
                        break
            print(f"{name}: {n} conversazioni", flush=True)
        except Exception as e:
            failed.append(f"{name} ({e!r})")

    def shuffled(repo, config=None):
        return load_dataset(repo, config, split="train").shuffle(seed=args.seed)

    def squad(repo, config, lang):
        ds = shuffled(repo, config)
        contexts = list({r["context"] for r in ds.select(range(min(5000, len(ds))))})
        previous = None
        for row in ds:
            k = rng.choice([0, 1, 2, 2])  # quante pagine "sbagliate" aggiungere
            distractors = [(None, c) for c in rng.sample(contexts, k) if c != row["context"]]
            # 1 esempio su 10: nessuna pagina contiene la risposta
            other = previous if previous and rng.random() < 0.1 else None
            yield from_squad(row, rng, lang, distractors, other_question=other)
            previous = row["question"]

    if "it" in args.lang:
        add("alpaca-gpt4-italian", lambda: (from_alpaca_gpt4(r) for r in shuffled("FreedomIntelligence/alpaca-gpt4-italian")))
        add("camoscio", lambda: (from_camoscio(r) for r in shuffled("teelinsan/camoscio")))
        add("squad_it (risposte dalle pagine web)", lambda: squad("crux82/squad_it", None, "it"))
    if "en" in args.lang:
        add("alpaca-gpt4 (inglese)", lambda: (from_camoscio(r) for r in shuffled("vicgalle/alpaca-gpt4")))
        add("squad (risposte dalle pagine web, inglese)", lambda: squad("rajpurkar/squad", "plain_text", "en"))

    oasst_rows = []

    def oasst(lang):
        if not oasst_rows:
            for split in ("train", "validation"):
                oasst_rows.extend(load_dataset("OpenAssistant/oasst2", split=split).select_columns(
                    ["message_id", "parent_id", "text", "role", "lang", "rank", "deleted", "review_result"]
                ).to_list())
        out = oasst_conversations(oasst_rows, lang)
        rng.shuffle(out)
        return out

    for lang in args.lang:
        add(f"oasst2 ({lang})", lambda lang=lang: oasst(lang))

    rng.shuffle(convs)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{args.out}: {len(convs)} conversazioni in totale")

    if failed:
        raise SystemExit("ATTENZIONE, fonti non scaricate:\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()
