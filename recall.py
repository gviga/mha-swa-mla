"""Experiment 2: synthetic long-range recall ("needle retrieval"), accuracy vs distance.

Each sequence of length L is random filler tokens, with one  KEY v  pair somewhere and a
QUERY as the last token. The model must output v at the QUERY position.  d = distance
between v and QUERY.  Chance accuracy = 1 / n_values.

    ... f f KEY v f f f ... f f QUERY  ->  v
              |<------- d ------->|

Full attention can retrieve v directly. With a sliding window (W=64) the query cannot see v
once d >= W, so the information must be *relayed* through intermediate tokens across layers
(theoretical reach n_layer * (W-1) = 252), which the model has to learn.

    python recall.py --attn mla_swa --seed 0
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

N_FILL, N_VAL = 32, 16
KEY, QUERY = N_FILL + N_VAL, N_FILL + N_VAL + 1
VOCAB = N_FILL + N_VAL + 2
EVAL_DISTANCES = [4, 16, 32, 48, 56, 63, 64, 72, 96, 128, 160, 192, 224, 252]


def make_batch(B, L, device, gen, dist=None):
    """Returns x (B, L), y (B, L) with loss only at the last position, and distances (B,)."""
    x = torch.randint(0, N_FILL, (B, L), generator=gen)
    vals = torch.randint(N_FILL, N_FILL + N_VAL, (B,), generator=gen)
    if dist is None:
        d = torch.randint(1, L - 2, (B,), generator=gen)       # value position >= 1
    else:
        d = torch.full((B,), dist)
    r = torch.arange(B)
    vpos = (L - 1) - d
    x[r, vpos - 1] = KEY
    x[r, vpos] = vals
    x[:, -1] = QUERY
    y = torch.full((B, L), -1)
    y[:, -1] = vals
    return x.to(device), y.to(device), d


@torch.no_grad()
def eval_by_distance(model, L, device, n=512):
    model.eval()
    gen, acc = torch.Generator().manual_seed(999), {}
    for d in EVAL_DISTANCES:
        x, y, _ = make_batch(n, L, device, gen, dist=d)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, _ = model(x)                                # last position only
        acc[d] = (logits[:, -1].argmax(-1) == y[:, -1]).float().mean().item()
    model.train()
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attn", default="mha", choices=["mha", "swa", "mla", "mla_swa"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--L", type=int, default=256)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--kv_latent", type=int, default=64)
    p.add_argument("--iters", type=int, default=2500)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="results/recall")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    cfg = GPTConfig(vocab_size=VOCAB, block_size=a.L, n_layer=4, n_head=4, n_embd=192, bias=False,
                    dropout=0.0, attn=a.attn, pos_emb="rope", window=a.window,
                    kv_latent=a.kv_latent, q_latent=96, rope_dim=16)
    model = GPT(cfg).to(device)
    opt = model.configure_optimizers(0.1, a.lr, (0.9, 0.99), "cuda" if device == "cuda" else "cpu")
    gen = torch.Generator().manual_seed(a.seed)
    curve, t0 = [], time.time()
    for it in range(a.iters + 1):
        if it % 500 == 0:
            acc = eval_by_distance(model, a.L, device, n=256)
            mean = sum(acc.values()) / len(acc)
            curve.append((it, mean))
            print(f"[{a.attn} s{a.seed}] it {it:5d} mean acc {mean:.3f}  ({time.time() - t0:.0f}s)")
        if it == a.iters:
            break
        warm = 200
        lr = a.lr * min(1.0, (it + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, it / a.iters)))
        for g in opt.param_groups:
            g["lr"] = max(lr, a.lr / 20)
        x, y, _ = make_batch(a.batch_size, a.L, device, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, _ = model(x)                                # (B, 1, V): last position only
        loss = F.cross_entropy(logits[:, -1].float(), y[:, -1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    acc = eval_by_distance(model, a.L, device, n=1024)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(args=vars(a), acc_by_distance=acc, curve=curve, chance=1 / N_VAL),
              open(os.path.join(a.out, f"{a.attn}_s{a.seed}.json"), "w"), indent=1)
    print(" ".join(f"{d}:{v:.2f}" for d, v in acc.items()))


if __name__ == "__main__":
    main()
