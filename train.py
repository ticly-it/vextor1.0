#!/usr/bin/env python3
"""Passo 3: addestra il modello.

Da zero, sul testo libero:
    python train.py --data out/pretrain --out out/pretrain-model --preset small --steps 100000

Fine-tuning sulle conversazioni, partendo dal modello pre-addestrato:
    python train.py --data out/chat --out out/chat-model --init-from out/pretrain-model/best.pt \
        --steps 5000 --lr 1e-4

Riprendere un addestramento interrotto:
    python train.py --data out/pretrain --out out/pretrain-model --resume
"""

import argparse
import math
import os
import time

import numpy as np
import torch

from milo.model import PRESETS, ModelConfig, Milo, load_checkpoint, pick_device, save_checkpoint
from milo.tokenizer import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="cartella con train.bin e val.bin")
    p.add_argument("--out", required=True, help="cartella dove salvare il modello")
    p.add_argument("--tokenizer", default="out/tokenizer.json")
    p.add_argument("--preset", choices=PRESETS, default="small", help="taglia del modello (solo da zero)")
    p.add_argument("--init-from", help="checkpoint da cui partire (fine-tuning)")
    p.add_argument("--resume", action="store_true", help="riprende da <out>/last.pt")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1, help="accumula N batch prima di aggiornare i pesi")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--compile", action="store_true", help="torch.compile (più veloce su GPU)")
    p.add_argument("--device", default=pick_device())
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = args.device
    tok = Tokenizer.load(args.tokenizer)
    last_path, best_path = os.path.join(args.out, "last.pt"), os.path.join(args.out, "best.pt")

    # ------------------------------------------------------------- modello
    step, best_val, opt_state = 0, float("inf"), None
    if args.resume:
        model, ckpt = load_checkpoint(last_path, device)
        step, best_val, opt_state = ckpt["step"], ckpt["best_val"], ckpt["optimizer"]
        print(f"Riprendo da {last_path} (step {step})")
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
    data = {
        name: np.memmap(os.path.join(args.data, f"{name}.bin"), dtype=np.uint16, mode="r")
        for name in ("train", "val")
    }
    block = cfg.context_size
    for name, arr in data.items():
        if len(arr) < 2:
            raise SystemExit(f"{name}.bin è vuoto")
        block = min(block, len(arr) - 1)
    if block < cfg.context_size:
        print(f"Attenzione: pochi dati, uso sequenze di {block} token invece di {cfg.context_size}")

    def get_batch(name):
        arr = data[name]
        starts = torch.randint(len(arr) - block, (args.batch_size,)).tolist()
        x = torch.from_numpy(np.stack([arr[i : i + block] for i in starts]).astype(np.int64))
        y = torch.from_numpy(np.stack([arr[i + 1 : i + 1 + block] for i in starts]).astype(np.int64))
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    # -------------------------------------------------------- ottimizzatore
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
        fused=device == "cuda",
    )
    if opt_state:
        optimizer.load_state_dict(opt_state)

    def lr_at(s):  # salita lineare, poi discesa a coseno fino al 10%
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        progress = min(1.0, (s - args.warmup) / max(1, args.steps - args.warmup))
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    use_amp = device == "cuda"
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp)  # noqa: E731
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    run_model = torch.compile(model) if args.compile else model

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

    def save(path):
        save_checkpoint(path, model, optimizer=optimizer.state_dict(), step=step, best_val=best_val)

    # ----------------------------------------------------------------- ciclo
    model.train()
    t0, running = time.time(), None
    try:
        while step < args.steps:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            for _ in range(args.grad_accum):
                x, y = get_batch("train")
                with autocast():
                    loss = run_model(x, y)[1]
                (loss / args.grad_accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            running = loss.item() if running is None else 0.95 * running + 0.05 * loss.item()
            if step % 50 == 0:
                dt = (time.time() - t0) / 50
                t0 = time.time()
                print(f"step {step:>7}/{args.steps}  loss {running:.3f}  lr {lr_at(step):.2e}  {dt * 1000:.0f} ms/step")

            if step % args.eval_every == 0 or step == args.steps:
                val = evaluate()
                marker = ""
                if val < best_val:
                    best_val = val
                    save(best_path)
                    marker = "  (migliore, salvato)"
                save(last_path)
                print(f"== step {step}: val loss {val:.3f}{marker}")
                t0 = time.time()
    except KeyboardInterrupt:
        save(last_path)
        print(f"\nInterrotto: stato salvato in {last_path}, riprendi con --resume")
        return

    print(f"Fine. Modello migliore: {best_path} (val loss {best_val:.3f})")


if __name__ == "__main__":
    main()
