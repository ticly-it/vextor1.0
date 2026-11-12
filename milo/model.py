"""Transformer decoder (stile GPT) scritto da zero in PyTorch.

Architettura: embedding dei token -> N blocchi [attenzione causale + MLP] con
pre-normalizzazione (RMSNorm), posizioni rotative (RoPE), pesi dell'embedding
condivisi con lo strato di uscita.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    context_size: int = 1024  # numero massimo di token che il modello vede
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0


# Taglie pronte. I parametri indicati sono approssimativi (vocabolario 16k).
PRESETS: dict[str, dict] = {
    # per provare che tutto funzioni, anche su CPU
    "tiny": dict(n_layer=4, n_head=4, n_embd=128, context_size=256),  # ~3M
    # addestrabile su una singola GPU consumer
    "small": dict(n_layer=8, n_head=8, n_embd=512, context_size=1024),  # ~30M
    # taglia GPT-2 small: serve una GPU seria e molti dati
    "base": dict(n_layer=12, n_head=12, n_embd=768, context_size=2048),  # ~100M
    # taglia GPT-2 medium: più GPU / giorni di training
    "large": dict(n_layer=24, n_head=16, n_embd=1024, context_size=2048),  # ~320M
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(head_dim: int, length: int, device=None):
    """Seni e coseni per le posizioni rotative, forma (length, head_dim/2)."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.outer(torch.arange(length, device=device).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Ruota le coppie di dimensioni di x (B, H, T, D) in base alla posizione."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is not None:
            # Generazione: si riusano chiavi/valori dei token già elaborati.
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        # Con la cache e un solo token nuovo non serve la maschera causale.
        causal = cache is None or T > 1
        if causal and k.size(2) != T:
            # prompt aggiunto a una cache esistente: maschera esplicita
            past = k.size(2) - T
            mask = torch.ones(T, k.size(2), dtype=torch.bool, device=x.device).tril(past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=causal, dropout_p=self.dropout if self.training else 0.0
            )
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin, cache))
        return x + self.drop(self.mlp(self.norm2(x)))


class Milo(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # pesi condivisi

        cos, sin = rope_tables(cfg.n_embd // cfg.n_head, cfg.context_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layer) ** 0.5)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, caches=None, pos: int = 0):
        """idx: (B, T) ID dei token. Con `targets` restituisce anche la loss.

        `caches`/`pos` servono solo in generazione: `pos` è la posizione del
        primo token di `idx` nella sequenza completa.
        """
        T = idx.size(1)
        if pos + T > self.cfg.context_size:
            raise ValueError(f"Sequenza più lunga del contesto ({self.cfg.context_size})")
        cos, sin = self.rope_cos[pos : pos + T], self.rope_sin[pos : pos + T]

        x = self.drop(self.embed(idx))
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None if caches is None else caches[i])
        x = self.norm(x)

        if targets is None:
            return self.head(x[:, -1:]), None
        logits = self.head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        repetition_penalty: float = 1.1,
        stop_ids: tuple[int, ...] = (),
    ):
        """Genera token uno alla volta (generatore). Si ferma su uno degli stop_ids."""
        self.eval()
        device = self.embed.weight.device
        room = self.cfg.context_size - len(prompt)
        if room <= 0:
            raise ValueError("Il prompt riempie tutto il contesto")
        max_new_tokens = min(max_new_tokens, room)

        caches = [{} for _ in self.blocks]
        idx = torch.tensor([prompt], dtype=torch.long, device=device)
        logits, _ = self(idx, caches=caches)
        pos = len(prompt)
        seen = set(prompt[-256:])

        for _ in range(max_new_tokens):
            logits = logits[0, -1].float()
            if repetition_penalty != 1.0 and seen:
                ids = torch.tensor(list(seen), device=device)
                vals = logits[ids]
                logits[ids] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)

            if temperature <= 0:
                next_id = int(logits.argmax())
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                    logits[logits < kth] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                if top_p and top_p < 1.0:
                    sorted_p, order = probs.sort(descending=True)
                    drop = sorted_p.cumsum(0) - sorted_p > top_p
                    sorted_p[drop] = 0.0
                    probs = torch.zeros_like(probs).scatter_(0, order, sorted_p)
                    probs /= probs.sum()
                next_id = int(torch.multinomial(probs, 1))

            if next_id in stop_ids:
                return
            yield next_id
            seen.add(next_id)
            logits, _ = self(torch.tensor([[next_id]], device=device), caches=caches, pos=pos)
            pos += 1


def save_checkpoint(path: str, model: Milo, **extra) -> None:
    raw = getattr(model, "_orig_mod", model)  # nel caso sia passato da torch.compile
    torch.save({"config": asdict(raw.cfg), "model": raw.state_dict(), **extra}, path)


def load_checkpoint(path: str, device: str = "cpu") -> tuple[Milo, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = Milo(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
