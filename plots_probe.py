"""Figures (figures/probe_*.png) and table (results/probe/summary.md) from probe.py's output.

    python plots_probe.py
"""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from plots import COLORS, LABELS, save, style, window_line  # noqa: E402  (same look as plots.py)

V4 = ["mha", "swa", "mla", "mla_swa"]
V5 = ["nanogpt"] + V4
P = "results/probe"
CMAP = "Blues"                                                    # sequential: one hue, light -> dark
NL = chr(10)
UNIFORM_SINK0 = np.mean([1 / (i + 1) for i in range(1, 64)])     # token-0 mass of a uniform head, 1 <= i < 64
LAYERS = [f"layer {l}" for l in range(4)]


def seeds(final, v):
    return [final[k] for k in sorted(final) if k.rsplit("_s", 1)[0] == v]


def agg(runs, fn):
    a = np.array([fn(r) for r in runs], dtype=float)
    return a.mean(0), (a.std(0, ddof=1) if len(a) > 1 else np.zeros_like(a.mean(0)))


def line(ax, x, runs, fn, v, **kw):
    m, s = agg(runs, fn)
    ax.plot(x, m, **{**style(v), **kw})
    ax.fill_between(x, m - s, m + s, color=COLORS[v], alpha=0.15, linewidth=0)


def bars(ax, final, fn, variants=V5):
    w = 0.8 / len(variants)
    for j, v in enumerate(variants):
        m, s = agg(seeds(final, v), fn)
        ax.bar(np.arange(4) + (j - (len(variants) - 1) / 2) * w, m, w * 0.88, yerr=s, color=COLORS[v],
               label=LABELS[v], error_kw=dict(lw=0.8, ecolor="#5f5e5a"))
    ax.set_xticks(range(4), LAYERS)


def fig_maps(z):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10.3))
    for r, v in enumerate(V4):
        A = z[v].astype(np.float32)
        for l in range(4):
            ax = axes[r, l]
            im = ax.imshow(A[l].mean(0) + 1e-6, cmap=CMAP, norm=LogNorm(1e-4, 1), interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            if r == 0:
                ax.set_title(LAYERS[l])
            if l == 0:
                ax.set_ylabel(LABELS[v], fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.5, label="attention probability (mean over heads, log scale)")
    fig.suptitle("Attention maps on one 256-character validation passage (query = row, key = column)", y=0.92)
    fig.savefig("figures/probe_attention_maps.png", bbox_inches="tight")
    plt.close(fig)


def fig_newline_head(final, z):
    text = str(z["text"])
    T0, T1 = 120, 216
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.9))
    for ax, v in zip(axes, V5):
        a = final[f"{v}_s0"]["attn"][1]
        h = int(np.argmax(a["nl_mass"]))
        ax.imshow(z[v][1, h].astype(np.float32)[T0:T1, T0:T1] + 1e-6, cmap=CMAP, norm=LogNorm(1e-3, 1),
                  interpolation="nearest")
        nl = [i - T0 for i in range(T0, T1) if text[i] == NL and text[i - 1] != NL]   # one label per "\n\n"
        ax.set_xticks(nl, ["\\n"] * len(nl), fontsize=7, color="#c2410c")
        ax.set_yticks([]), ax.grid(False)
        ax.set_title(f"{LABELS[v]}\nL1 head {h}: {a['nl_mass'][h]:.0%} of mass on newlines,\n"
                     f"newline ||v|| = {a['nl_vnorm_ratio'][h]:.2f}x other keys", fontsize=8)
    axes[0].set_ylabel("query (characters 120-215)")
    fig.suptitle("The newline sink: each dark column is a newline that the following queries keep attending to "
                 "(log colour, seed 0)", y=1.03)
    save(fig, "probe_newline_head")


def fig_distance(final):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
    d = np.arange(129)
    for l, ax in enumerate(axes):
        for v in V5:
            line(ax, d, seeds(final, v), lambda r: np.mean(r["attn"][l]["dist_profile"], 0) + 1e-7, v,
                 marker=None, linewidth=1.5)
        ax.set_yscale("log"), ax.set_ylim(1e-5, 1)
        ax.set_title(LAYERS[l]), ax.set_xlabel("key distance i − j")
        window_line(ax, 63, "W−1 = 63")
    axes[0].set_ylabel("mean attention (over heads)")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle("Attention vs relative distance (queries i ≥ 128): windowed models drop to exactly 0 past "
                 "W−1; full models keep a long, thin tail")
    save(fig, "probe_attention_distance")


def fig_entropy(final):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    pos = np.arange(256)
    ax = axes[0]
    for v in V5:
        line(ax, pos, seeds(final, v), lambda r: np.mean([np.mean(a["entropy_by_pos"], 0) for a in r["attn"]], 0),
             v, marker=None, linewidth=1.5)
    ax.plot(pos[1:], np.log(pos[1:] + 1), color="#8a8984", linewidth=1, linestyle=":", label="uniform over all keys")
    window_line(ax, 64)
    ax.set(xlabel="query position i", ylabel="attention entropy (nats)\nmean over layers & heads", ylim=(0, 3.5),
           title="Entropy vs position")
    ax.legend(fontsize=7, loc="lower right")
    bars(axes[1], final, lambda r: [np.mean(a["eff_keys_late"]) for a in r["attn"]])
    axes[1].set(ylabel="exp(H), queries i ≥ 64, mean over heads", title="Effective number of keys attended")
    save(fig, "probe_entropy")


def fig_sinks(final):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    panels = [("sink0_early", "max over heads: mean attention\non token 0 (queries 1 ≤ i < 64)",
               "No positional (first-token) sink"),
              ("nl_mass", "max over heads: attention mass\non newline keys (queries i ≥ 64)", "A sink on '\\n' in layer 1"),
              ("nl_enrich", "max over heads: newline mass /\nnewline share of visible keys", "…at 10–18× its frequency")]
    for ax, (key, ylab, title) in zip(axes, panels):
        bars(ax, final, lambda r: [max(a[key]) for a in r["attn"]])
        ax.set(ylabel=ylab, title=title)
    axes[0].axhline(UNIFORM_SINK0, color="#5f5e5a", linestyle=":", linewidth=1)
    axes[0].axhline(0.3, color="#5f5e5a", linestyle="--", linewidth=1)
    axes[0].text(3.45, UNIFORM_SINK0, "uniform head", va="bottom", ha="right", fontsize=7, color="#5f5e5a")
    axes[0].text(3.45, 0.3, "sink threshold ε = 0.3", va="bottom", ha="right", fontsize=7, color="#5f5e5a")
    axes[0].set_ylim(0, 0.4)
    axes[2].axhline(1, color="#5f5e5a", linestyle=":", linewidth=1)
    axes[1].legend(fontsize=7)
    save(fig, "probe_sinks")


def fig_hidden(final):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    depth = np.arange(5)
    for ax, (key, lab) in zip(axes, [("eff_rank", "effective rank exp H(ρ)  (of C = 192)"),
                                     ("anisotropy", "anisotropy (mean pairwise cosine)"),
                                     ("pos0_norm_ratio", "‖h‖ at position 0 / median ‖h‖")]):
        for v in V5:
            line(ax, depth, seeds(final, v), lambda r: [h[key] for h in r["hidden"]], v)
        ax.set_xticks(depth, ["emb"] + [f"L{l}" for l in range(4)]), ax.set_title(lab, fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle("Residual-stream geometry by depth (8,192 validation tokens; ρ = token covariance / trace)")
    save(fig, "probe_hidden_geometry")


def fig_kv(final):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    ax = axes[0]
    for v in V4:
        line(ax, np.arange(4), seeds(final, v), lambda r: [c["eff_rank"] for c in r["cache"]], v)
    ax.set_xticks(range(4), LAYERS)
    ax.set(title="Effective rank of what is cached", ylabel="exp H(ρ)")
    ax.text(0.02, 0.97, "MHA / SWA: [K | V], 384 dims\nMLA / hybrid: latent c_kv, 64 dims", transform=ax.transAxes,
            va="top", fontsize=7, color="#5f5e5a")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    for v in V4:
        r = seeds(final, v)[0]
        for l, ls in [(1, ":"), (3, "-")]:
            spec = np.cumsum(r["cache"][l]["spectrum"])
            ax.plot(np.arange(1, 65), spec, color=COLORS[v], linestyle=ls, linewidth=1.5, label=f"{LABELS[v]}, layer {l}")
    ax.set(xlabel="# principal directions", ylabel="cumulative variance", xscale="log",
           title="Spectrum (seed 0): MHA's 384-dim cache lives in ≈ 64 dims")
    ax.legend(fontsize=6, ncol=2)
    save(fig, "probe_kv_rank")


def summary(final):
    L = ["# Probe summary (auto-generated by plots_probe.py)", "",
         "Experiment-1 checkpoints, mean over 3 seeds, fixed batch of 32 x 256 validation characters. "
         "'Late' = queries i ≥ 64.", "",
         "| variant | attention beyond 64 tokens (mean / max head) | eff. keys (late) | max token-0 mass (i < 64) "
         "| newline-head mass | newline enrichment | newline ‖v‖ / other ‖v‖ | cached eff. rank / dims | "
         "cache variance in top 64 dims |", "|---|---|---|---|---|---|---|---|---|"]
    for v in V5:
        rs = seeds(final, v)

        def m(fn):
            return np.mean([fn(r) for r in rs])

        def head(r, k):                                   # layer-1 head with the most newline mass
            a = r["attn"][1]
            return a[k][int(np.argmax(a["nl_mass"]))]
        L.append(f"| {LABELS[v]} "
                 f"| {m(lambda r: np.mean([a['far_mass_late'] for a in r['attn']])):.3f} / "
                 f"{m(lambda r: max(max(a['far_mass_late']) for a in r['attn'])):.3f} "
                 f"| {m(lambda r: np.mean([a['eff_keys_late'] for a in r['attn']])):.1f} "
                 f"| {m(lambda r: max(max(a['sink0_early']) for a in r['attn'])):.3f} (uniform {UNIFORM_SINK0:.3f}) "
                 f"| {m(lambda r: head(r, 'nl_mass')):.2f} | {m(lambda r: head(r, 'nl_enrich')):.1f}x "
                 f"| {m(lambda r: head(r, 'nl_vnorm_ratio')):.2f} "
                 f"| {m(lambda r: np.mean([c['eff_rank'] for c in r['cache']])):.1f} / {rs[0]['cache'][0]['dim']} "
                 f"| {m(lambda r: min(c['var_top64'] for c in r['cache'])):.3f} (worst layer) |")
    open(f"{P}/summary.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    final = json.load(open(f"{P}/final.json"))
    z = np.load(f"{P}/maps.npz")
    fig_maps(z)
    fig_newline_head(final, z)
    fig_distance(final)
    fig_entropy(final)
    fig_sinks(final)
    fig_hidden(final)
    fig_kv(final)
    summary(final)
    print(f"wrote {P}/summary.md and figures/probe_*.png")
