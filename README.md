# Sliding-window and latent attention in nanoGPT

This project adds three attention variants to [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) and compares them with standard attention at small scale:

* **SWA**, Sliding Window Attention, from [Mistral 7B](https://arxiv.org/abs/2310.06825);
* **MLA**, Multi-head Latent Attention, from [DeepSeek-V2](https://arxiv.org/abs/2405.04434), Section 2.1;
* **a hybrid**, which is MLA with a sliding window.

**Start with [REPORT.md](REPORT.md).** It explains the methods, how the code was checked, which analyses I chose and why, what held up from each paper's claims, and the advantages and disadvantages of each method.

## What changed from nanoGPT

I changed only two of nanoGPT's files, and every change is marked `[hybrid-attn]`. [nanogpt.diff](nanogpt.diff) shows the complete diff against upstream commit `3adf61e`, and the original files are in [upstream_ref/](upstream_ref/).

* [model.py](model.py) contains almost all the new code: RoPE, the sliding window (a simple mask and a faster block version, selected with `window_impl`), the KV cache with its ring buffer, MLA, and a `probe()` method on each attention class that exposes the attention matrix for analysis.
* [train.py](train.py) has the new configuration options, a configurable seed, and saves final metrics to JSON.

With the default settings the model behaves exactly like upstream nanoGPT, and a test checks this. `configurator.py`, `sample.py`, `data/shakespeare_char/prepare.py` and `LICENSE` are unchanged from upstream.

## Files I added

| file | what it is for |
|---|---|
| [tests/test_attention.py](tests/test_attention.py) | correctness tests (43, about 10 s on CPU) |
| [config/shakespeare_hybrid.py](config/shakespeare_hybrid.py) | the small model used here (4 layers, 1.7M parameters), sized for a 4 GB GPU |
| [recall.py](recall.py) | the long-range recall task, which tests how far back a window can reach |
| [bench_attention.py](bench_attention.py) | cache memory, generation time and training time |
| [probe.py](probe.py) | looks inside the trained models: attention maps, sinks, rank of the cache, hidden states |
| [plots.py](plots.py), [plots_probe.py](plots_probe.py) | produce the figures and summary tables |
| [scripts/](scripts/) | shell scripts that run the experiments |

The results are already included, so nothing needs to be run to read the report. [figures/](figures/) holds every plot in the report, and [results/](results/) holds the raw JSON for each run. For a compact view of the numbers, see [results/summary.md](results/summary.md) and [results/probe/summary.md](results/probe/summary.md).

The trained checkpoints are not included because of their size. `probe.py` and `sample.py` need them, so run `scripts/run_shakespeare.sh` first if you want to use those two scripts.

## Running the code

```bash
pip install -r requirements.txt
python -m pytest -q tests        # correctness tests
bash scripts/run_all.sh          # everything; about 2.5 hours on an RTX 2050 (4 GB)
```

To train one model, run one experiment, or generate text:

```bash
python data/shakespeare_char/prepare.py
python train.py config/shakespeare_hybrid.py --attn=mla_swa --seed=0 --out_dir=out/shakespeare/mla_swa_s0
python recall.py --attn=mla_swa --seed=0
python bench_attention.py
python sample.py --out_dir=out/shakespeare/mla_swa_s0
```

`--attn` can be `mha`, `swa`, `mla` or `mla_swa`. Add `--window_impl=block` to train a windowed model with the faster window. The config sets `compile=False` because `torch.compile` needs Triton, which is not available on native Windows.
