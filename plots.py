"""Make figures (figures/*.png) and summary tables (results/summary.md) from results/*.

    python plots.py
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# validated categorical palette (fixed order) + neutral gray for the vanilla nanoGPT reference
COLORS = {"mha": "#2a78d6", "swa": "#eb6834", "mla": "#1baf7a", "mla_swa": "#eda100", "nanogpt": "#8a8984"}
MARKERS = {"mha": "o", "swa": "s", "mla": "^", "mla_swa": "D", "nanogpt": "x"}
LABELS = {"mha": "MHA (RoPE)", "swa": "SWA", "mla": "MLA", "mla_swa": "MLA+SWA (hybrid)",
          "nanogpt": "nanoGPT vanilla (learned pos.)"}
ORDER = ["nanogpt", "mha", "swa", "mla", "mla_swa"]
W = 64
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "font.size": 9, "legend.frameon": False,
                     "lines.linewidth": 2, "lines.markersize": 5})


def style(key, **kw):
    return dict(color=COLORS[key], marker=MARKERS[key], label=LABELS[key],
                linestyle="--" if key == "nanogpt" else "-", **kw)


def window_line(ax, x=W, text="W = 64", y=1.0, ha="left"):
    ax.axvline(x, color="#5f5e5a", linewidth=1, linestyle=":")
    ax.text(x, y, f" {text} " , transform=ax.get_xaxis_transform(), va="top", ha=ha, fontsize=8, color="#5f5e5a",
            bbox=dict(facecolor="white", edgecolor="none", pad=1))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", bbox_inches="tight")
    plt.close(fig)


def mean_std(vals):
    a = np.array(vals, dtype=float)
    return a.mean(0), (a.std(0, ddof=1) if len(a) > 1 else np.zeros_like(a.mean(0)))


def shakespeare(lines):
    runs = defaultdict(list)
    for f in glob.glob("results/shakespeare/*.json"):
        key = os.path.basename(f).rsplit("_s", 1)[0]
        runs[key].append(json.load(open(f)))
    if not runs:
        return
    keys = [k for k in ORDER if k in runs]
    base = np.array([r["final_val_loss"] for r in runs["mha"]]).mean()
    lines += ["## Experiment 1 - Tiny Shakespeare (char-level, T=256, 3000 iters)", "",
              "| variant | seeds | val loss (mean ± std) | Δ vs MHA | params | attn params | "
              "cache / token / layer | ms / iter | peak MB |", "|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        rs = runs[k]
        v = np.array([r["final_val_loss"] for r in rs])
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        lines.append(f"| {LABELS[k]} | {len(rs)} | {v.mean():.4f} ± {sd:.4f} | {v.mean() - base:+.4f} | "
                     f"{rs[0]['params'] / 1e6:.3f}M | {rs[0]['attn_params'] / 1e3:.0f}K | "
                     f"{rs[0]['cache_elems_per_token_per_layer']} | "
                     f"{np.mean([r['ms_per_iter'] for r in rs]):.1f} | {np.mean([r['peak_mem_mb'] for r in rs]):.0f} |")
    lines.append("")

    # per-position loss difference vs MHA
    ref = np.array([r["per_position_loss"] for r in runs["mha"]]).mean(0)
    k_sm = 9
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k in [k for k in keys if k not in ("mha", "nanogpt")]:  # nanoGPT: different pos. scheme, see table
        d = np.array([r["per_position_loss"] for r in runs[k]]).mean(0) - ref
        ds = np.convolve(d, np.ones(k_sm) / k_sm, mode="valid")
        ax.plot(np.arange(len(ds)) + k_sm // 2, ds, **style(k, markevery=24))
    ax.axhline(0, color="#2a78d6", linewidth=1)
    window_line(ax)
    ax.set(xlabel="position in context (tokens)", ylabel="loss − MHA loss (nats/char)",
           title="Where the variants lose quality: per-position loss vs MHA")
    ax.legend()
    save(fig, "shakespeare_per_position")

    # per-position loss table: early vs late in context
    lines += ["Per-position validation loss (mean over seeds), early vs late in the context:", "",
              "| variant | pos 0-63 (< W) | pos 64-255 (≥ W) |", "|---|---|---|"]
    for k in keys:
        pl = np.array([r["per_position_loss"] for r in runs[k]]).mean(0)
        lines.append(f"| {LABELS[k]} | {pl[:W].mean():.4f} | {pl[W:].mean():.4f} |")
    lines.append("")


def recall(lines, src="results/recall", small_src="results/recall_w16", fname="recall_accuracy",
           title="Long-range recall (1 fact): accuracy vs distance"):
    runs = defaultdict(list)
    for f in glob.glob(f"{src}/*.json"):
        runs[os.path.basename(f).rsplit("_s", 1)[0]].append(json.load(open(f)))
    if not runs:
        return
    keys = [k for k in ORDER if k in runs]
    ds = [int(d) for d in runs[keys[0]][0]["acc_by_distance"]]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    lines += [f"## {title} (chance = 0.0625)", "",
              "| variant | " + " | ".join(f"d={d}" for d in ds) + " |", "|---" * (len(ds) + 1) + "|"]
    for k in keys:
        m, s = mean_std([[r["acc_by_distance"][str(d)] for d in ds] for r in runs[k]])
        ax.plot(ds, m, **style(k))
        ax.fill_between(ds, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=COLORS[k], alpha=0.15, linewidth=0)
        lines.append(f"| {LABELS[k]} ({len(runs[k])} seeds) | " + " | ".join(f"{v:.2f}" for v in m) + " |")
    small = defaultdict(list)  # extra runs with W=16: receptive field 4*(16-1) = 60 tokens
    for f in glob.glob(f"{small_src}/*.json") if small_src else []:
        small[os.path.basename(f).rsplit("_s", 1)[0]].append(json.load(open(f)))
    for k in [k for k in ORDER if k in small]:
        m, _ = mean_std([[r["acc_by_distance"][str(d)] for d in ds] for r in small[k]])
        st = style(k)
        st.update(label=LABELS[k] + ", W = 16", linestyle=":", markerfacecolor="white")
        ax.plot(ds, m, **st)
        lines.append(f"| {LABELS[k]}, W=16 ({len(small[k])} seed) | " + " | ".join(f"{v:.2f}" for v in m) + " |")
    if small:
        window_line(ax, 60, "reach of W=16 = 4 x 15 = 60 tokens", y=0.75, ha="left")
    ax.axhline(runs[keys[0]][0]["chance"], color="#8a8984", linewidth=1, linestyle="--")
    ax.text(ds[-1], runs[keys[0]][0]["chance"], "chance ", ha="right", va="bottom", fontsize=8, color="#5f5e5a")
    window_line(ax)
    ax.set(xlabel="distance between value and query (tokens)", ylabel="accuracy", ylim=(0, 1.05),
           title=title)
    ax.legend(loc="lower left", fontsize=7)
    save(fig, fname)
    lines.append("")


def bench(lines):
    if not os.path.exists("results/bench/bench.json"):
        return
    b = json.load(open("results/bench/bench.json"))
    lines += [f"## Experiment 3 - systems benchmarks ({b['device']}, bf16, random weights)", ""]

    # cache memory
    cm = b["cache_memory_bytes_B1"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k in ["mha", "swa", "mla", "mla_swa"]:
        T, by = zip(*cm[k])
        ax.plot(T, np.array(by) / 2**20, **style(k))
    ax.set(xscale="log", yscale="log", xlabel="context length (tokens)", ylabel="KV cache (MiB, batch 1)",
           title="KV-cache memory vs context (measured = formula)")
    ax.legend()
    save(fig, "cache_memory")
    Ts = [r[0] for r in cm["mha"]]
    lines += ["KV-cache size (batch 1, 4 layers, bf16), measured allocation:", "",
              "| variant | " + " | ".join(f"T={t}" for t in Ts) + " | ratio vs MHA @ max T |",
              "|---" * (len(Ts) + 2) + "|"]
    for k in ["mha", "swa", "mla", "mla_swa"]:
        by = [r[1] for r in cm[k]]
        lines.append(f"| {LABELS[k]} | " + " | ".join(f"{x / 2**10:.0f} KiB" for x in by) +
                     f" | {cm['mha'][-1][1] / by[-1]:.1f}x |")
    lines.append("")

    # decode latency
    for B in (16,):
        dl = b[f"decode_ms_per_token_B{B}"]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for name, rows in dl.items():
            c, t = zip(*rows)
            ax.plot(c, t, **style(name))
        ax.set(xscale="log", xlabel="tokens already in context", ylabel="ms per generated token",
               title=f"Decode latency vs context (batch {B})")
        ax.legend(fontsize=7)
        save(fig, f"decode_latency_B{B}")
        ctxs = [r[0] for r in next(iter(dl.values()))]
        lines += [f"Decode latency, ms per token (batch {B}):", "",
                  "| variant | " + " | ".join(f"ctx={c}" for c in ctxs) + " |", "|---" * (len(ctxs) + 1) + "|"]
        lines += [f"| {n} | " + " | ".join(f"{t:.2f}" for _, t in rows) + " |" for n, rows in dl.items()]
        lines.append("")

    # training cost: naive (mask) vs optimized (block) window; peak memory is in the table
    tr = b["train_step_B8"]
    order = ["mha", "swa", "swa(block)", "mla", "mla_swa", "mla_swa(block)"]

    def label(k):
        return LABELS[k.split("(")[0]] + (", block (optimized)" if "block" in k else ", mask (naive)" if "swa" in k else "")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k in order:
        st = style(k.split("(")[0])
        st.update(label=label(k))
        if "block" in k:
            st.update(linestyle=":", markerfacecolor="white")
        rows = [r for r in tr[k] if r[1] is not None]
        ax.plot([r[0] for r in rows], [r[1] for r in rows], **st)
    ax.set(xscale="log", yscale="log", xlabel="training sequence length T", ylabel="ms per training step",
           title="Training step time (batch 8, W = 64)")
    ax.legend(fontsize=7)
    save(fig, "train_step_time")
    Ts = [r[0] for r in tr["mha"]]
    lines += ["Training step (fwd+bwd+AdamW, batch 8): ms / peak MiB (OOM = out of memory on 4 GB):", "",
              "| variant | " + " | ".join(f"T={t}" for t in Ts) + " |", "|---" * (len(Ts) + 1) + "|"]
    for k in order:
        cells = [("OOM" if r[1] is None else f"{r[1]:.0f} ms / {r[2]:.0f}") for r in tr[k]]
        lines.append(f"| {label(k)} | " + " | ".join(cells) + " |")
    lines.append("")

    ps = b["paper_scale_cache_MiB_per_sequence"]
    Ts = list(ps)
    lines += ["Cache per sequence at paper scale (formula, bf16, MiB):", "",
              "| model | " + " | ".join(f"T={t}" for t in Ts) + " |", "|---" * (len(Ts) + 1) + "|"]
    for name in ps[Ts[0]]:
        lines.append(f"| {name} | " + " | ".join(f"{ps[t][name]:,.0f}" for t in Ts) + " |")
    lines.append("")


if __name__ == "__main__":
    out = ["# Results summary (auto-generated by plots.py)", ""]
    shakespeare(out)
    recall(out)
    bench(out)
    open("results/summary.md", "w", encoding="utf-8").write("\n".join(out))
    print("wrote results/summary.md and figures/*.png")
