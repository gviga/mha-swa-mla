"""Correctness tests for the attention variants.   Run:  python -m pytest -q tests"""
import importlib.util
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import GPT, GPTConfig, causal_attention, sliding_window_mask  # noqa: E402

ATTN_TYPES = ("mha", "swa", "mla", "mla_swa")
V = 50


def make(attn, **kw):
    torch.manual_seed(0)
    base = dict(vocab_size=V, block_size=64, n_layer=2, n_head=4, n_embd=64, bias=False,
                window=8, kv_latent=32, q_latent=48, rope_dim=8, attn=attn, pos_emb="rope")
    base.update(kw)
    return GPT(GPTConfig(**base)).double().eval()


def tokens(B=2, T=40, seed=1):
    return torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(seed))


def logits(m, x):
    return m(x, all_logits=True)[0]


def perturb(x, i):
    x = x.clone()
    x[:, i] = (x[:, i] + 1) % V
    return x


def test_default_config_is_upstream_nanogpt():
    """With attn='mha', pos_emb='learned' our model must equal untouched upstream nanoGPT."""
    spec = importlib.util.spec_from_file_location("upstream", os.path.join(ROOT, "upstream_ref", "model.py"))
    up = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(up)
    kw = dict(vocab_size=V, block_size=64, n_layer=2, n_head=4, n_embd=64)
    ref = up.GPT(up.GPTConfig(**kw)).double().eval()
    ours = GPT(GPTConfig(**kw)).double().eval()
    ours.load_state_dict(ref.state_dict(), strict=False)  # upstream has an unused 'attn.bias' buffer only without flash
    x = tokens()
    assert torch.equal(ref(x, x)[0], ours(x, x)[0])


def test_mask_has_exactly_W_keys():
    m = sliding_window_mask(20, 5, "cpu")
    assert m.sum(1).tolist() == [1, 2, 3, 4] + [5] * 16
    assert m[10, 6] and not m[10, 5] and not m[10, 11]


@pytest.mark.parametrize("attn", ATTN_TYPES)
def test_causality(attn):
    """Changing a future token must not change earlier logits."""
    m, x = make(attn), tokens()
    y1, y2 = logits(m, x), logits(m, perturb(x, 25))
    assert torch.allclose(y1[:, :25], y2[:, :25])
    assert not torch.allclose(y1[:, 25:], y2[:, 25:])


@pytest.mark.parametrize("attn", ["swa", "mla_swa"])
def test_window_boundary_single_layer(attn):
    """With ONE layer, output at i depends on token i-W+1 but not on token i-W."""
    W, i = 8, 30
    m, x = make(attn, n_layer=1, window=W), tokens()
    y = logits(m, x)[:, i]
    assert torch.allclose(y, logits(m, perturb(x, i - W))[:, i])          # just outside
    assert not torch.allclose(y, logits(m, perturb(x, i - W + 1))[:, i])  # oldest token inside


@pytest.mark.parametrize("attn", ["swa", "mla_swa"])
def test_receptive_field_grows_with_depth(attn):
    """With L layers information can travel at most L*(W-1) positions back."""
    W, L, i = 8, 2, 35
    m, x = make(attn, n_layer=L, window=W), tokens()
    y = logits(m, x)[:, i]
    far = i - L * (W - 1)
    assert not torch.allclose(y, logits(m, perturb(x, far))[:, i])
    assert torch.allclose(y, logits(m, perturb(x, far - 1))[:, i])


@pytest.mark.parametrize("pair", [("mha", "swa"), ("mla", "mla_swa")])
def test_large_window_equals_full_attention(pair):
    full, win = make(pair[0]), make(pair[1], window=1000)
    win.load_state_dict(full.state_dict())
    x = tokens()
    assert torch.allclose(logits(full, x), logits(win, x))


@pytest.mark.parametrize("attn", ATTN_TYPES)
@pytest.mark.parametrize("prefill", [3, 20])
def test_cached_decode_matches_full_forward(attn, prefill):
    """Prefill + token-by-token decode must reproduce the full forward pass.
    T=40 > W=8, so the rolling buffer wraps several times; prefill=20 > W also exercises
    the rolling-prefill path. For MLA this also proves absorbed decode == re-expanded K, V."""
    m, x = make(attn), tokens()
    ref = logits(m, x)
    T = x.size(1)
    m.setup_cache(x.size(0), T)
    outs = [m(x[:, :prefill], start_pos=0, all_logits=True)[0]]
    for t in range(prefill, T):
        outs.append(m(x[:, t:t + 1], start_pos=t)[0])
    got = torch.cat(outs, dim=1)
    assert torch.allclose(ref, got, atol=1e-10), (ref - got).abs().max()


def test_cached_decode_learned_positions():
    """KV cache also works for the vanilla nanoGPT baseline (learned absolute positions)."""
    m, x = make("mha", pos_emb="learned"), tokens()
    ref = logits(m, x)
    m.setup_cache(x.size(0), x.size(1))
    outs = [m(x[:, :5], start_pos=0, all_logits=True)[0]]
    outs += [m(x[:, t:t + 1], start_pos=t)[0] for t in range(5, x.size(1))]
    assert torch.allclose(ref, torch.cat(outs, 1), atol=1e-10)


@pytest.mark.parametrize("attn", ATTN_TYPES)
def test_cache_size_matches_formula(attn):
    m = make(attn)
    cfg, B, T = m.config, 3, 40
    m.setup_cache(B, T)
    tokens_held = min(T, cfg.window) if cfg.windowed else T
    expected = B * cfg.n_layer * tokens_held * cfg.cache_elems_per_token_per_layer() * 8  # float64
    assert m.cache_nbytes() == expected


@pytest.mark.parametrize("attn", ATTN_TYPES)
def test_probe_reproduces_attention_output(attn):
    """The explicit probabilities used by probe.py are exactly the ones SDPA applies:
    c_proj(probs @ v) == attn(x), rows sum to 1, and nothing is visible outside the (window) mask."""
    m = make(attn)
    a = m.transformer.h[0].attn
    x = torch.randn(2, 40, 64, dtype=torch.double, generator=torch.Generator().manual_seed(3))
    pos = torch.arange(40)
    out = a.probe(x, pos)
    p, v = out["probs"].double(), out["v"]
    y = a.c_proj((p @ v).transpose(1, 2).reshape(2, 40, 64))
    assert torch.allclose(y, a(x, pos), atol=1e-6)
    assert torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)))
    allowed = sliding_window_mask(40, 8, x.device) if attn in ("swa", "mla_swa") else torch.ones(40, 40, dtype=torch.bool).tril()
    assert (p[..., ~allowed] == 0).all()


@pytest.mark.parametrize("W,T", [(1, 5), (3, 10), (8, 9), (8, 16), (8, 24), (8, 29), (8, 8), (8, 5)])
def test_block_window_equals_mask(W, T):
    """The optimized window gives the mask's result: exact blocks, a partial last block, W >= T."""
    g = torch.Generator().manual_seed(W + T)
    q, k, v = (torch.randn(2, 3, T, 16, dtype=torch.double, generator=g) for _ in range(3))
    assert torch.allclose(causal_attention(q, k, v, W, impl="mask"), causal_attention(q, k, v, W, impl="block"), atol=1e-10)


@pytest.mark.parametrize("attn", ["swa", "mla_swa"])
@pytest.mark.parametrize("T", [9, 24, 37])
def test_block_window_full_model(attn, T):
    """Same logits, loss and gradients through the whole model (window = 8)."""
    x = tokens(T=T)
    mask, block = make(attn, window_impl="mask"), make(attn, window_impl="block")
    l_mask, l_block = mask(x, x)[1], block(x, x)[1]
    assert torch.allclose(l_mask, l_block, atol=1e-10)
    l_mask.backward(), l_block.backward()
    for p, r in zip(mask.parameters(), block.parameters()):
        assert torch.allclose(p.grad, r.grad, atol=1e-9)
