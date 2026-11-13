#!/usr/bin/env python3
"""Addestra Milo 1.

Pre-addestramento da zero, con una durata in ore (i passi si calcolano da soli
misurando la velocità della GPU):
    python train.py --data out/pretrain --out out/pretrain-model --preset base --hours 48

Fine-tuning sulle conversazioni, partendo dal modello pre-addestrato:
    python train.py --data out/chat --out out/chat-model --init-from out/pretrain-model/best.pt \\
        --epochs 2 --lr 1e-4

Riprendere un addestramento interrotto (stesso comando con --resume):
    python train.py --data out/pretrain --out out/pretrain-model --resume

La cartella dei dati contiene uno o più file train*.bin e val*.bin (token a 16 bit);
se accanto a un file X.bin c'è X.mask, la loss conta solo i token con maschera 1.
"""

import argparse
import glob
import math
import os
import shutil
import time

import numpy as np
import torch

from milo.model import PRESETS, ModelConfig, Milo, load_checkpoint, pick_device, save_checkpoint
from milo.tokenizer import Tokenizer


class Shards:
    """Tutti i file train*.bin (o val*.bin) di una cartella, letti senza caricarli in memoria."""

    def __init__(self, folder: str, split: str, block: int):
        paths = sorted(glob.glob(os.path.join(folder, f"{split}*.bin")))
        self.tokens, self.masks = [], []
        for p in paths:
            arr = np.memmap(p, dtype=np.uint16, mode="r") if os.path.getsize(p) else np.zeros(0, np.uint16)
            if len(arr) <= block + 1:
                continue
            mask_path = p[: -len(".bin")] + ".mask"
            self.tokens.append(arr)
            self.masks.append(np.memmap(mask_path, dtype=np.uint8, mode="r") if os.path.exists(mask_path) else None)
        if not self.tokens:
            raise SystemExit(f"Nessun file {split}*.bin utilizzabile in {folder} (vuoti o troppo corti)")
        self.block = block
        sizes = np.array([len(a) - block - 1 for a in self.tokens], dtype=np.float64)
        self.weights = sizes / sizes.sum()
        self.total = int(sum(len(a) for a in self.tokens))

    def sample(self, rng: np.random.Generator):
        while True:
            k = rng.choice(len(self.tokens), p=self.weights) if len(self.tokens) > 1 else 0
            arr, mask = self.tokens[k], self.masks[k]
            i = int(rng.integers(0, len(arr) - self.block - 1))
            x = arr[i : i + self.block].astype(np.int64)
            y = arr[i + 1 : i + 1 + self.block].astype(np.int64)
            if mask is not None:
                m = mask[i + 1 : i + 1 + self.block]
                if not m.any():
                    continue  # finestra senza risposte da imparare: se ne prende un'altra
                y = np.where(m == 1, y, -1)
            return x, y


def copy_local(src: str, dst: str) -> str:
    """Copia i dati su disco locale (su Colab leggere a caso da Google Drive è lentissimo)."""
    os.makedirs(dst, exist_ok=True)
    for p in glob.glob(os.path.join(src, "*.bin")) + glob.glob(os.path.join(src, "*.mask")):
        q = os.path.join(dst, os.path.basename(p))
        if not os.path.exists(q) or os.path.getsize(q) != os.path.getsize(p):
            print(f"  copio {os.path.basename(p)} ({os.path.getsize(p) / 1e9:.2f} GB)", flush=True)
            shutil.copyfile(p, q + ".tmp")
            os.replace(q + ".tmp", q)
    return dst


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="cartella con train*.bin e val*.bin")
    p.add_argument("--out", required=True, help="cartella dove salvare il modello")
    p.add_argument("--tokenizer", default="out/tokenizer.json")
    p.add_argument("--preset", choices=PRESETS, default="small", help="taglia del modello (solo da zero)")
    p.add_argument("--init-from", help="checkpoint da cui partire (fine-tuning)")
    p.add_argument("--resume", action="store_true", help="riprende da <out>/last.pt")
    durata = p.add_argument_group("durata (una sola; con --resume vale quella salvata)")
    durata.add_argument("--steps", type=int, help="numero di passi")
    durata.add_argument("--hours", type=float, help="ore di addestramento: i passi si calcolano dalla velocità misurata")
    durata.add_argument("--epochs", type=float, help="quante volte vedere i dati (utile per il fine-tuning)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1, help="accumula N batch prima di aggiornare i pesi")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=0, help="0 = automatico (circa 100 verifiche in tutto)")
    p.add_argument("--eval-batches", type=int, default=40)
    p.add_argument("--save-every-min", type=float, default=20, help="salva last.pt ogni N minuti")
    p.add_argument("--sample-prompt", default="L'Italia è", help="testo da completare a ogni verifica")
    p.add_argument("--local-copy", help="copia prima i dati in questa cartella locale (consigliato su Colab)")
    p.add_argument("--compile", action="store_true", help="torch.compile (più veloce su GPU recenti)")
    p.add_argument("--device", default=pick_device())
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if sum(x is not None for x in (args.steps, args.hours, args.epochs)) > 1:
        p.error("usa una sola tra --steps, --hours e --epochs")

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = args.device
    tok = Tokenizer.load(args.tokenizer)
    last_path, best_path = os.path.join(args.out, "last.pt"), os.path.join(args.out, "best.pt")

    # ------------------------------------------------------------- modello
    step, best_val, total_steps, ckpt = 0, float("inf"), None, None
    if args.resume:
        if not os.path.exists(last_path):
            raise SystemExit(f"{last_path} non esiste: togli --resume per iniziare")
        model, ckpt = load_checkpoint(last_path, device)
        step, best_val = ckpt["step"], ckpt["best_val"]
        total_steps = args.steps or ckpt.get("total_steps")
        print(f"Riprendo da {last_path}: passo {step} di {total_steps}")
    elif args.init_from:
        model, _ = load_checkpoint(args.init_from, device)
        print(f"Parto dai pesi di {args.init_from}")
    else:
        cfg = ModelConfig(vocab_size=tok.vocab_size, dropout=args.dropout, **PRESETS[args.preset])
        model = Milo(cfg).to(device)
    if model.cfg.vocab_size != tok.vocab_size:
        raise SystemExit("Il tokenizer non corrisponde a quello usato per questo modello")
    cfg = model.cfg
    print(f"Modello: {model.num_params() / 1e6:.1f}M parametri, contesto {cfg.context_size}, device {device}")

    # ---------------------------------------------------------------- dati
    data_dir = copy_local(args.data, args.local_copy) if args.local_copy else args.data
    block = cfg.context_size
    train, val = Shards(data_dir, "train", block), Shards(data_dir, "val", block)
    rng = np.random.default_rng(args.seed + step)
    tokens_per_step = args.batch_size * args.grad_accum * block
    print(f"Dati: {train.total / 1e6:,.0f}M token di addestramento, {val.total / 1e6:,.1f}M di verifica "
          f"| {tokens_per_step:,} token per passo")

    if total_steps is None:
        if args.steps:
            total_steps = args.steps
        elif args.epochs:
            total_steps = max(1, int(args.epochs * train.total / tokens_per_step))
        elif not args.hours:
            total_steps = 20000
    measuring = total_steps is None  # con --hours si misura la velocità nei primi passi

    def get_batch(split):
        src = train if split == "train" else val
        xs, ys = zip(*(src.sample(rng) for _ in range(args.batch_size)))
        x = torch.from_numpy(np.stack(xs)).to(device, non_blocking=True)
        y = torch.from_numpy(np.stack(ys)).to(device, non_blocking=True)
        return x, y

    # -------------------------------------------------------- ottimizzatore
    decay = [q for q in model.parameters() if q.dim() >= 2]
    no_decay = [q for q in model.parameters() if q.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=device == "cuda",
    )
    # Precisione mista su GPU: bfloat16 dove supportato, altrimenti float16
    # (es. le T4 di Colab), che richiede di riscalare i gradienti.
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
    ckpt = None  # libera memoria
    autocast = lambda: torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp)  # noqa: E731
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    run_model = torch.compile(model) if args.compile else model

    def lr_at(s):  # salita lineare, poi discesa a coseno fino al 10%
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        if total_steps is None:
            return args.lr
        progress = min(1.0, (s - args.warmup) / max(1, total_steps - args.warmup))
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for _ in range(args.eval_batches):
            x, y = get_batch("val")
            with autocast():
                losses.append(run_model(x, y)[1].item())
        model.train()
        return sum(losses) / len(losses)

    @torch.no_grad()
    def sample_text():
        if not args.sample_prompt:
            return ""
        prompt = [tok.token_id("<|bos|>")] + tok.encode(args.sample_prompt)
        out = list(model.generate(prompt, max_new_tokens=40, temperature=0.8, stop_ids=tuple(tok.special.values())))
        model.train()
        return (args.sample_prompt + tok.decode(out)).replace("\n", " ")

    def save(path, full=True):
        extra = {"step": step, "best_val": best_val, "total_steps": total_steps}
        if full:
            extra |= {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}
        save_checkpoint(path + ".tmp", model, **extra)
        os.replace(path + ".tmp", path)  # mai un file a metà, anche se la sessione cade

    # ----------------------------------------------------------------- ciclo
    model.train()
    eval_every = args.eval_every
    t_start, t_log, t_save = time.time(), time.time(), time.time()
    measure_from, running = None, None
    try:
        while total_steps is None or step < total_steps:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            for _ in range(args.grad_accum):
                x, y = get_batch("train")
                with autocast():
                    loss = run_model(x, y)[1]
                scaler.scale(loss / args.grad_accum).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()

            if measuring:
                if step == 10:
                    if device == "cuda":
                        torch.cuda.synchronize()
                    measure_from = time.time()
                elif step == 60:
                    if device == "cuda":
                        torch.cuda.synchronize()
                    per_step = (time.time() - measure_from) / 50
                    # ~4% del tempo va in verifiche e salvataggi
                    total_steps = max(step + 1, int(args.hours * 3600 * 0.96 / per_step))
                    measuring = False
                    print(f"Velocità: {per_step:.2f} s/passo, {tokens_per_step / per_step:,.0f} token/s -> "
                          f"{total_steps:,} passi in {args.hours:g} ore "
                          f"({total_steps * tokens_per_step / 1e9:.2f} miliardi di token, "
                          f"{total_steps * tokens_per_step / train.total:.1f} passaggi sui dati)", flush=True)

            if not eval_every and total_steps:
                eval_every = max(100, total_steps // 100)

            if step % 50 == 0:
                dt = (time.time() - t_log) / 50
                t_log = time.time()
                eta = f"  fine tra {(total_steps - step) * dt / 3600:.1f} h" if total_steps else ""
                print(f"passo {step:>7}/{total_steps or '?'}  loss {running:.3f}  lr {lr_at(step):.2e}  "
                      f"{tokens_per_step / dt:,.0f} token/s{eta}", flush=True)

            last = total_steps is not None and step >= total_steps
            if eval_every and (step % eval_every == 0 or last):
                v = evaluate()
                marker = ""
                if v < best_val:
                    best_val = v
                    save(best_path, full=False)
                    marker = "  (migliore, salvato)"
                save(last_path)
                t_save = time.time()
                example = sample_text()
                print(f"== passo {step}: val loss {v:.3f}{marker}", flush=True)
                if example:
                    print(f"   esempio: {example}", flush=True)
                t_log = time.time()
            elif time.time() - t_save > args.save_every_min * 60:
                save(last_path)
                t_save = t_log = time.time()
    except KeyboardInterrupt:
        save(last_path)
        print(f"\nInterrotto: stato salvato in {last_path}, riprendi con --resume")
        return

    hours = (time.time() - t_start) / 3600
    print(f"Fine in {hours:.1f} h. Modello migliore: {best_path} (val loss {best_val:.3f})")


if __name__ == "__main__":
    main()
