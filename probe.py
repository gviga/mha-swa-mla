"""Experiment 4: look inside the trained Shakespeare models (checkpoints of Experiment 1).

For every checkpoint, on a fixed batch of 32 x 256 validation characters, it measures:
  * attention maps per head, and attention vs relative distance i - j  -> visual check that the window
    is exactly W wide in the trained models, and how much attention full models put beyond W;
  * attention entropy H_i = -sum_j p_ij log p_ij  and  exp(H) = "effective number of keys";
  * attention sinks: mass on the first token (StreamingLLM, Xiao et al. 2023; threshold 0.3 as in
    Gu et al. 2024) and mass on newline keys, plus the value norm of those keys (sinks are no-op keys);
  * residual-stream geometry per layer: spectral entropy of the token covariance, effective rank,
    anisotropy, norm of position 0;
  * effective rank of what each variant caches: MHA's [K | V] (384 dims) vs MLA's latent c_kv (64 dims).

    python probe.py      -> results/probe/final.json, results/probe/maps.npz   (figures: plots_probe.py)
"""
import contextlib
import glob
import io
import json
import math
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
W_REF = 64          # = W. "late" queries i >= 64 see exactly W keys under SWA; "early" = 1 <= i < 64
NEWLINE = 0         # id of '\n' in shakespeare_char (the vocabulary is sorted)
OUT = "results/probe"


def load_model(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    with contextlib.redirect_stdout(io.StringIO()):          # GPT() prints its parameter count
        m = GPT(GPTConfig(**ck["model_args"]))
    m.load_state_dict({k.removeprefix("_orig_mod."): v.float() for k, v in ck["model"].items()})
    return m.to(DEVICE).eval()


def val_batch(n_seq=32, T=256, seed=1234):
    data = np.memmap("data/shakespeare_char/val.bin", dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - T - 1, (n_seq,), generator=torch.Generator().manual_seed(seed))
    x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + T].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def example_text(T=256):
    """One readable validation passage starting at a line break, for the attention-map figures."""
    data = np.memmap("data/shakespeare_char/val.bin", dtype=np.uint16, mode="r")
    itos = pickle.load(open("data/shakespeare_char/meta.pkl", "rb"))["itos"]
    start = 20000 + int(np.argmax(data[20000:21000] == NEWLINE)) + 1
    ids = data[start:start + T].astype(np.int64)
    return torch.from_numpy(ids)[None].to(DEVICE), "".join(itos[int(i)] for i in ids)


@torch.no_grad()
def capture(model, idx):
    """One forward pass. Per layer: the attention probe (model.py, `probe`) and the residual stream.
    hidden[0] = embeddings, hidden[l + 1] = output of block l."""
    attn, hidden = [], []
    hooks = [model.transformer.drop.register_forward_hook(lambda m, i, o: hidden.append(o))]
    for b in model.transformer.h:
        hooks.append(b.attn.register_forward_pre_hook(lambda m, args: attn.append(m.probe(args[0], args[1]))))
        hooks.append(b.register_forward_hook(lambda m, i, o: hidden.append(o)))
    try:
        logits, _ = model(idx, all_logits=True)
    finally:
        for h in hooks:
            h.remove()
    return logits, attn, hidden


def attention_stats(P, v, idx, window):
    """P (B, H, T, T) attention of one layer, v (B, H, T, hs) its values. Per-head statistics."""
    B, H, T, _ = P.shape
    i = torch.arange(T, device=P.device)
    n_vis = (i + 1) if window is None else (i + 1).clamp(max=window)
    late, early = i >= W_REF, (i >= 1) & (i < W_REF)
    rel = i[:, None] - i[None, :]                                           # i - j
    ent = -(P * P.clamp_min(1e-30).log()).sum(-1)                           # (B, H, T)
    q0 = T // 2                                                             # profile over queries i >= T/2
    profile = torch.stack([torch.diagonal(P, -d, -2, -1)[..., q0 - d:].mean((0, 2)) for d in range(q0 + 1)], -1)
    # newline keys: share of attention they get vs the share a uniform head would give them
    vis = torch.ones(T, T, device=P.device).tril()
    if window is not None:
        vis = vis.triu(-(window - 1))
    nl = (idx == NEWLINE).float()                                           # (B, T)
    nl_mass = torch.einsum("bhij,bj->bhi", P, nl)[..., late].mean((0, 2))
    nl_unif = torch.einsum("ij,bj->bi", (vis / n_vis[:, None])[late], nl).mean()
    vn = v.norm(dim=-1)                                                     # (B, H, T)
    isnl = idx == NEWLINE
    return dict(
        entropy_late=ent[..., late].mean((0, 2)),
        eff_keys_late=ent[..., late].exp().mean((0, 2)),
        entropy_by_pos=ent.mean(0),                                         # (H, T)
        dist_late=(P * rel).sum(-1)[..., late].mean((0, 2)),
        far_mass_late=(P * (rel >= W_REF)).sum(-1)[..., late].mean((0, 2)),  # attention beyond W tokens
        dist_profile=profile,                                               # (H, T/2 + 1)
        sink0_early=P[:, :, early, 0].mean((0, 2)),
        nl_mass=nl_mass, nl_enrich=nl_mass / nl_unif,
        nl_vnorm_ratio=torch.stack([vn[:, h][isnl].mean() / vn[:, h][~isnl].mean() for h in range(H)]),
    )


def spectrum_stats(X):
    """X (N, D). Von Neumann entropy of rho = Cov / tr Cov, its exp (effective rank), top-k variance."""
    X = X.double() - X.double().mean(0)
    lam = torch.linalg.eigvalsh(X.T @ X / X.size(0)).clamp_min(0).flip(0)
    p = lam / lam.sum()
    Hs = -(p * p.clamp_min(1e-300).log()).sum()
    cum = p.cumsum(0)
    return {"spec_entropy": (Hs / math.log(X.size(1))).item(), "eff_rank": Hs.exp().item(), "dim": X.size(1),
            "var_top16": cum[15].item(), "var_top64": cum[63].item(), "spectrum": p[:64].tolist()}


def hidden_stats(h):
    """h (B, T, C) residual stream at one depth."""
    X = h.reshape(-1, h.size(-1)).float()
    out = spectrum_stats(X)
    s = F.normalize(X.double(), dim=-1).sum(0)
    out["anisotropy"] = ((s @ s - X.size(0)) / (X.size(0) * (X.size(0) - 1))).item()   # mean pairwise cosine
    norms = h.float().norm(dim=-1)
    out["pos0_norm_ratio"] = (norms[:, 0].mean() / norms[:, 1:].median()).item()
    return out


def cache_stats(probe):
    """What each variant caches per token: MLA the latent c_kv (64); MHA/SWA [K before RoPE | V] (384)."""
    if "c_kv" in probe:
        return spectrum_stats(probe["c_kv"].reshape(-1, probe["c_kv"].size(-1)))
    B, H, T, hs = probe["v"].shape
    return spectrum_stats(torch.cat([probe["k_nope"], probe["v"]], -1).transpose(1, 2).reshape(B * T, 2 * H * hs))


def tolist(d):
    return {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in d.items()}


@torch.no_grad()
def analyse(model, x, y):
    window = model.config.window if model.config.windowed else None
    logits, attn, hidden = capture(model, x)
    return dict(val_loss=F.cross_entropy(logits.flatten(0, 1), y.flatten()).item(),
                attn=[tolist(attention_stats(a["probs"], a["v"], x, window)) for a in attn],
                hidden=[hidden_stats(h) for h in hidden],
                cache=[cache_stats(a) for a in attn])


def main():
    os.makedirs(OUT, exist_ok=True)
    x, y = val_batch()
    res = {}
    for f in sorted(glob.glob("out/shakespeare/*_s*/ckpt.pt")):
        name = os.path.basename(os.path.dirname(f))
        res[name] = analyse(load_model(f), x, y)
        print(f"{name:12s} val loss {res[name]['val_loss']:.4f}")
    json.dump(res, open(f"{OUT}/final.json", "w"))
    idx, text = example_text()                    # attention maps of seed 0 on one readable passage
    maps = {"text": np.array(text)}
    for v in ["nanogpt", "mha", "swa", "mla", "mla_swa"]:
        _, attn, _ = capture(load_model(f"out/shakespeare/{v}_s0/ckpt.pt"), idx)
        maps[v] = torch.stack([a["probs"][0] for a in attn]).half().cpu().numpy()   # (L, H, T, T)
    np.savez_compressed(f"{OUT}/maps.npz", **maps)


if __name__ == "__main__":
    main()
