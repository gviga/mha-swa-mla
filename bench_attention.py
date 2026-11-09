"""Experiment 3: systems benchmarks (no training needed, random weights).

  (a) KV-cache memory vs context length  -- measured allocation, checked against the formula
  (b) per-token decode latency vs context length
  (c) training step time & peak memory vs sequence length (shows the cost of the mask-based SWA)
  (d) the same cache formulas evaluated at Mistral-7B / DeepSeek-V2 scale, for perspective

    python bench_attention.py            -> results/bench/bench.json
"""
import json
import os
import time

import torch

from model import GPT, GPTConfig

DEV = "cuda"
BASE = dict(vocab_size=65, block_size=256, n_layer=4, n_head=4, n_embd=192, bias=False, dropout=0.0,
            pos_emb="rope", window=64, kv_latent=64, q_latent=96, rope_dim=16)
VARIANTS = ["mha", "swa", "mla", "mla_swa"]
TRAIN_LENGTHS = [256, 512, 1024, 2048, 4096, 8192]


def build(attn, **kw):
    torch.manual_seed(0)
    return GPT(GPTConfig(**{**BASE, "attn": attn, **kw})).to(DEV).to(torch.bfloat16).eval()


def cache_memory(contexts):
    out = {}
    for attn in ["mha", "swa", "mla", "mla_swa"]:
        m = build(attn)
        rows = []
        for T in contexts:
            m.setup_cache(1, T)
            held = min(T, m.config.window) if m.config.windowed else T
            formula = m.config.n_layer * held * m.config.cache_elems_per_token_per_layer() * 2  # bf16
            assert m.cache_nbytes() == formula
            rows.append((T, m.cache_nbytes()))
            m.clear_cache()
        out[attn] = rows
        del m
    return out


@torch.no_grad()
def decode_latency(contexts, B, n_steps=32, reps=9):
    """ms per generated token. Variants are timed INTERLEAVED (every repetition runs all variants
    back-to-back) and we report the median, because a laptop GPU changes clock state mid-run."""
    models = {a: build(a) for a in VARIANTS}
    tok = torch.zeros(B, 1, dtype=torch.long, device=DEV)
    out = {n: [] for n in models}

    def run(m, ctx, steps):
        # Pretend `ctx` tokens are already cached: timing does not depend on cache contents,
        # and this avoids a (ctx x ctx) prefill on a 4 GB GPU.
        for blk in m.transformer.h:
            blk.attn.cache.length = ctx
        for i in range(steps):
            m(tok, start_pos=ctx + i)

    for ctx in contexts:
        for m in models.values():
            m.setup_cache(B, ctx + n_steps + 1)
            run(m, ctx, 4)  # warm-up
        times = {n: [] for n in models}
        for _ in range(reps):
            for n, m in models.items():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                run(m, ctx, n_steps)
                torch.cuda.synchronize()
                times[n].append(1000 * (time.perf_counter() - t0) / n_steps)
        for n, m in models.items():
            out[n].append((ctx, float(sorted(times[n])[reps // 2])))
            m.clear_cache()
        torch.cuda.empty_cache()
        print(f"B={B} ctx={ctx}: " + ", ".join(f"{n} {out[n][-1][1]:.2f}" for n in models))
    return out


def train_step_cost(seq_lens, B):
    # variants are interleaved at each T: a laptop GPU changes clock state during a long run
    variants = [("mha", "mask"), ("swa", "mask"), ("swa", "block"), ("mla", "mask"), ("mla_swa", "mask"), ("mla_swa", "block")]
    out = {f"{a}(block)" if i == "block" else a: [] for a, i in variants}
    for T in seq_lens:
        for attn, impl in variants:
            rows = out[f"{attn}(block)" if impl == "block" else attn]
            torch.manual_seed(0)
            m = GPT(GPTConfig(**{**BASE, "attn": attn, "window_impl": impl, "block_size": T})).to(DEV).train()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            x = torch.randint(0, 65, (B, T), device=DEV)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            times = []
            try:
                for i in range(13):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, loss = m(x, x)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    torch.cuda.synchronize()
                    if i >= 3:
                        times.append(time.perf_counter() - t0)
                rows.append((T, 1000 * sorted(times)[len(times) // 2], torch.cuda.max_memory_allocated() / 2**20))
            except torch.OutOfMemoryError:
                rows.append((T, None, None))
            del m, opt
            torch.cuda.empty_cache()
            print(attn, impl, rows[-1])
    return out


def paper_scale():
    """Cache per token (all layers, 2 bytes/elem) for model shapes from the papers."""
    def mib(elems):
        return elems * 2 / 2**20
    mistral = dict(layers=32, kv_heads=8, hs=128, W=4096)            # Mistral 7B uses GQA (8 KV heads)
    dsv2 = dict(layers=60, heads=128, hs=128, dc=512, dR=64)          # DeepSeek-V2
    rows = {}
    for T in [4096, 32768, 131072]:
        rows[T] = {
            "Mistral-7B full (GQA)": mib(T * 2 * mistral["kv_heads"] * mistral["hs"] * mistral["layers"]),
            "Mistral-7B SWA (GQA)": mib(min(T, mistral["W"]) * 2 * mistral["kv_heads"] * mistral["hs"] * mistral["layers"]),
            "DeepSeek-V2-shape MHA": mib(T * 2 * dsv2["heads"] * dsv2["hs"] * dsv2["layers"]),
            "DeepSeek-V2 MLA": mib(T * (dsv2["dc"] + dsv2["dR"]) * dsv2["layers"]),
            "DeepSeek-V2 MLA + SWA(W=4096)": mib(min(T, 4096) * (dsv2["dc"] + dsv2["dR"]) * dsv2["layers"]),
        }
    return rows


def main():
    os.makedirs("results/bench", exist_ok=True)
    res = dict(device=torch.cuda.get_device_name(0), config=BASE)
    res["cache_memory_bytes_B1"] = cache_memory([256, 1024, 4096, 16384, 65536])
    print("cache memory ok (matches formula)")
    res["decode_ms_per_token_B16"] = decode_latency([256, 1024, 4096, 16384], B=16)
    res["train_step_B8"] = train_step_cost(TRAIN_LENGTHS, B=8)
    res["paper_scale_cache_MiB_per_sequence"] = paper_scale()
    json.dump(res, open("results/bench/bench.json", "w"), indent=1)


if __name__ == "__main__":
    main()
