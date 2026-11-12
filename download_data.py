#!/usr/bin/env python3
"""Passo 0: scarica i dati per addestrare Milo 1 (serve `pip install datasets`).

    python download_data.py --out dati --wiki-mb 500

Crea nella cartella di destinazione:
    wikipedia.txt   testo italiano per il pre-addestramento
    chat.jsonl      conversazioni per il fine-tuning: domande e risposte,
                    risposte basate su una pagina (per la ricerca web)
                    e conversazioni sull'identità di Milo 1

Fonti (Hugging Face): wikimedia/wikipedia (italiano), teelinsan/camoscio
(istruzioni e risposte in italiano), crux82/squad_it (domande su un testo).
Controlla le licenze di ciascuna prima di un uso commerciale.
"""

import argparse
import json
import os
import random

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


def from_squad(row, rng, other_question=None):
    """Riga {context, question, answers} -> conversazione con blocco "context".

    Il contesto ha lo stesso formato che produce milo/search.py:
        [1] titolo \\n url \\n testo
    Con `other_question` si crea un esempio in cui la pagina non contiene la risposta.
    """
    context = (row.get("context") or "").strip()
    question = (row.get("question") or "").strip()
    answers = (row.get("answers") or {}).get("text") or []
    if not context or not question or not answers:
        return None
    title = (row.get("title") or "").replace("_", " ")
    url = "https://it.wikipedia.org/wiki/" + title.replace(" ", "_") if title else "https://it.wikipedia.org"
    title = title or "Wikipedia"
    block = f"[1] {title}\n{url}\n{context}"
    if other_question:
        return conversation(("context", block), ("user", other_question), ("assistant", rng.choice(NO_ANSWER)))
    answer = answers[0].strip()
    answer = answer[0].upper() + answer[1:]
    return conversation(("context", block), ("user", question), ("assistant", f"{answer} [1]."))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="dati")
    p.add_argument("--wiki-mb", type=int, default=500, help="MB di testo di Wikipedia da scaricare (0 = salta)")
    p.add_argument("--max-chat", type=int, default=60000, help="massimo di conversazioni da ogni fonte")
    p.add_argument("--identity-copies", type=int, default=40, help="ripetizioni delle conversazioni sull'identità")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    failed = []

    # ------------------------------------------------------------ Wikipedia
    if args.wiki_mb > 0:
        path = os.path.join(args.out, "wikipedia.txt")
        limit = args.wiki_mb * 1_000_000
        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.it", split="train", streaming=True)
            ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
            written = 0
            with open(path, "w", encoding="utf-8") as f:
                for row in ds:
                    text = (row.get("text") or "").strip()
                    if len(text) < 500:  # voci troppo corte: quasi solo elenchi
                        continue
                    written += f.write(f"{row.get('title', '')}\n\n{text}\n\n\n")
                    if written >= limit:
                        break
            print(f"wikipedia.txt: {written / 1e6:.0f} MB")
        except Exception as e:
            failed.append(f"Wikipedia ({e!r})")

    # ------------------------------------------------------- conversazioni
    convs = identity_conversations(rng, args.identity_copies)
    print(f"identità: {len(convs)} conversazioni")

    try:
        n = 0
        for row in load_dataset("teelinsan/camoscio", split="train").shuffle(seed=args.seed):
            c = from_camoscio(row)
            if c:
                convs.append(c)
                n += 1
                if n >= args.max_chat:
                    break
        print(f"domande e risposte: {n} conversazioni")
    except Exception as e:
        failed.append(f"camoscio ({e!r})")

    try:
        n = 0
        previous = None
        for row in load_dataset("crux82/squad_it", split="train").shuffle(seed=args.seed):
            # 1 esempio su 10: la pagina non contiene la risposta alla domanda
            other = previous if previous and rng.random() < 0.1 else None
            c = from_squad(row, rng, other_question=other)
            previous = (row.get("question") or "").strip() or previous
            if c:
                convs.append(c)
                n += 1
                if n >= args.max_chat:
                    break
        print(f"risposte da una pagina: {n} conversazioni")
    except Exception as e:
        failed.append(f"squad_it ({e!r})")

    rng.shuffle(convs)
    with open(os.path.join(args.out, "chat.jsonl"), "w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"chat.jsonl: {len(convs)} conversazioni in totale")

    if failed:
        raise SystemExit("ATTENZIONE, fonti non scaricate:\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()
