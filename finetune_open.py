#!/usr/bin/env python3
"""Trasforma un modello aperto già addestrato in Milo 1 (fine-tuning LoRA).

    pip install transformers peft accelerate bitsandbytes
    python build_chat.py --out dati/chat-aperto.jsonl --max-chat 1500 --identity-copies 8 \\
        --base-model "Gemma 3" --base-maker Google
    python finetune_open.py --data dati/chat-aperto.jsonl --out out/milo-aperto

    python chat.py --open-model out/milo-aperto

Il modello di base non viene modificato: si addestrano piccoli pesi aggiuntivi (LoRA) che gli
insegnano l'identità di Milo 1, lo stile e l'uso dei risultati della ricerca web. La loss conta
solo le risposte dell'assistente. Si può interrompere e riprendere con --resume.

Modelli di base provati nel formato: unsloth/gemma-3-12b-it (default, 12 miliardi di parametri,
contesto 128k, serve una GPU da 24 GB), unsloth/gemma-3-4b-it (va su una T4 da 16 GB).
I modelli Gemma hanno una licenza propria (Gemma Terms of Use): leggila prima di un uso commerciale.
"""

import argparse
import json
import math
import os
import random
import shutil
import time

import torch

from milo.open_model import SYSTEM_PROMPT, load_base, to_chat

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode_example(tok, messages, max_len):
    """Conversazione -> (ids, labels): labels = -100 ovunque tranne che sulle risposte dell'assistente."""
    chat = to_chat(messages)
    ids = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=False, return_dict=True)["input_ids"]
    labels = [-100] * len(ids)
    for k, m in enumerate(chat):
        if m["role"] != "assistant":
            continue
        before = tok.apply_chat_template(chat[:k], tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
        upto = tok.apply_chat_template(chat[: k + 1], tokenize=True, add_generation_prompt=False, return_dict=True)["input_ids"]
        if ids[: len(before)] != before or ids[: len(upto)] != upto:
            return ids[:max_len], ids[:max_len]  # template non "a prefissi": si impara tutto
        labels[len(before) : len(upto)] = ids[len(before) : len(upto)]
    if len(ids) > max_len:
        # troppo lungo: si tiene la fine (lì c'è la risposta), il contesto web viene accorciato in testa
        ids, labels = ids[:1] + ids[-(max_len - 1):], labels[:1] + labels[-(max_len - 1):]
    return ids, labels


def batches(examples, batch_size, rng):
    """Lotti di esempi di lunghezza simile (meno padding), in ordine casuale."""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]) + rng.random())
    groups = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(groups)
    return groups


def collate(examples, idx, pad_id, device):
    n = max(len(examples[i][0]) for i in idx)
    ids = torch.full((len(idx), n), pad_id, dtype=torch.long)
    labels = torch.full((len(idx), n), -100, dtype=torch.long)
    mask = torch.zeros((len(idx), n), dtype=torch.long)
    for r, i in enumerate(idx):
        x, y = examples[i]
        ids[r, : len(x)] = torch.tensor(x)
        labels[r, : len(y)] = torch.tensor(y)
        mask[r, : len(x)] = 1
    return ids.to(device), mask.to(device), labels.to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="unsloth/gemma-3-12b-it", help="modello di base su Hugging Face")
    p.add_argument("--data", required=True, help="conversazioni .jsonl (da build_chat.py)")
    p.add_argument("--out", required=True, help="cartella dell'adattatore")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16, help="dimensione dell'adattatore LoRA")
    p.add_argument("--max-len", type=int, default=2048, help="token massimi per conversazione in addestramento")
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--save-every-min", type=float, default=15)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-4bit", action="store_true", help="carica il modello a precisione piena (serve molta più memoria)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    state_path = os.path.join(args.out, "stato.pt")
    if args.resume and not os.path.exists(state_path):
        raise SystemExit(f"{state_path} non esiste: togli --resume per iniziare")

    # --------------------------------------------------------------- modello
    cuda = torch.cuda.is_available()
    four_bit = cuda and not args.no_4bit
    print(f"Carico {args.base} ({'4 bit' if four_bit else 'precisione piena'})...", flush=True)
    tok, model = load_base(args.base, four_bit)
    if four_bit:
        model = prepare_model_for_kbit_training(model)
    if cuda:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if args.resume:
        model = PeftModel.from_pretrained(model, args.out, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS,
            exclude_modules=r".*(vision_tower|vision_model|multi_modal_projector).*",  # solo la parte di testo
        ))
    device = next(model.parameters()).device
    trainable = [q for q in model.parameters() if q.requires_grad]
    total = sum(q.numel() for q in model.parameters())
    print(f"Parametri: {total / 1e9:.2f} miliardi, di cui addestrati {sum(q.numel() for q in trainable) / 1e6:.1f} milioni", flush=True)

    # ------------------------------------------------------------------ dati
    with open(args.data, encoding="utf-8") as f:
        convs = [json.loads(line)["messages"] for line in f if line.strip()]
    examples = [encode_example(tok, c, args.max_len) for c in convs]
    examples = [e for e in examples if any(y != -100 for y in e[1])]
    rng.shuffle(examples)  # stesso seme -> stessa divisione anche quando si riprende
    n_val = max(1, min(200, int(len(examples) * args.val_fraction)))
    val, train = examples[:n_val], examples[n_val:]
    steps_per_epoch = math.ceil(math.ceil(len(train) / args.batch_size) / args.grad_accum)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    print(f"Conversazioni: {len(train)} per l'addestramento, {len(val)} di verifica | {total_steps} passi", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    step = 0
    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        print(f"Riprendo dal passo {step}", flush=True)
    warmup = max(1, min(50, total_steps // 10))

    def lr_at(s):
        if s < warmup:
            return args.lr * (s + 1) / warmup
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total_steps - warmup))))

    amp_dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    autocast = lambda: torch.autocast("cuda", dtype=amp_dtype, enabled=cuda)  # noqa: E731

    def save():
        """Salva adattatore + stato in una cartella temporanea, poi la sostituisce a quella vera."""
        tmp = args.out.rstrip("/") + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained(tmp)
        torch.save({"optimizer": optimizer.state_dict(), "step": step}, os.path.join(tmp, "stato.pt"))
        with open(os.path.join(tmp, "milo.json"), "w", encoding="utf-8") as f:
            json.dump({"base": args.base, "system": SYSTEM_PROMPT, "step": step, "total_steps": total_steps}, f,
                      ensure_ascii=False, indent=1)
        old = args.out.rstrip("/") + ".old"
        shutil.rmtree(old, ignore_errors=True)
        if os.path.exists(args.out):
            os.rename(args.out, old)
        os.rename(tmp, args.out)
        shutil.rmtree(old, ignore_errors=True)

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for i in range(0, len(val), args.batch_size):
            ids, mask, labels = collate(val, list(range(i, min(i + args.batch_size, len(val)))), tok.pad_token_id, device)
            with autocast():
                losses.append(model(input_ids=ids, attention_mask=mask, labels=labels).loss.item())
        chat = to_chat([{"role": "user", "content": "Ciao, chi sei?"}])
        inputs = tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
        out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")
        model.train()
        return sum(losses) / len(losses), text

    # ----------------------------------------------------------------- ciclo
    model.train()
    eval_every = max(1, total_steps // 8)
    micro_per_epoch = math.ceil(len(train) / args.batch_size)
    t_log = t_save = time.time()
    running = None
    try:
        while step < total_steps:
            epoch = (step * args.grad_accum) // micro_per_epoch
            groups = batches(train, args.batch_size, random.Random(args.seed + epoch))
            pos = (step * args.grad_accum) % micro_per_epoch
            for group_i in range(pos, len(groups), args.grad_accum):
                for group in groups[group_i : group_i + args.grad_accum]:
                    ids, mask, labels = collate(train, group, tok.pad_token_id, device)
                    with autocast():
                        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                    (loss / args.grad_accum).backward()
                for g in optimizer.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                running = loss.item() if running is None else 0.9 * running + 0.1 * loss.item()

                if step % 10 == 0 or step == total_steps:
                    dt = (time.time() - t_log) / 10
                    t_log = time.time()
                    print(f"passo {step}/{total_steps}  loss {running:.3f}  fine tra {(total_steps - step) * dt / 60:.0f} min", flush=True)
                if step % eval_every == 0 or step == total_steps:
                    v, text = evaluate()
                    save()
                    t_save = t_log = time.time()
                    print(f"== passo {step}: val loss {v:.3f}\n   Ciao, chi sei? -> {text}", flush=True)
                elif time.time() - t_save > args.save_every_min * 60:
                    save()
                    t_save = t_log = time.time()
                if step >= total_steps:
                    break
    except KeyboardInterrupt:
        save()
        print(f"\nInterrotto: stato salvato in {args.out}, riprendi con --resume")
        return
    print(f"Fine. Adattatore salvato in {args.out}")


if __name__ == "__main__":
    main()
