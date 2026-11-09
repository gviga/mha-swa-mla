"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

[hybrid-attn] Modified from karpathy/nanoGPT @ 3adf61e to support four attention variants:
    attn='mha'      standard causal multi-head attention (upstream nanoGPT)
    attn='swa'      MHA restricted to a sliding window of W tokens   (Mistral 7B, arXiv:2310.06825)
    attn='mla'      Multi-head Latent Attention + decoupled RoPE      (DeepSeek-V2, arXiv:2405.04434, Sec. 2.1)
    attn='mla_swa'  hybrid: MLA whose latent KV cache is a rolling buffer of W tokens
plus RoPE positions (pos_emb='rope') and a KV cache for incremental decoding.
With the defaults (attn='mha', pos_emb='learned') the model is identical to upstream nanoGPT.

Window semantics: query i sees keys j with i-W < j <= i, i.e. exactly W keys including
itself. (The Mistral text says "i-W .. i" = W+1 tokens; its figure shows W. We use W.)
Shape legend: B batch, T tokens, H heads, hs head size, dc kv latent, dq query latent,
dR decoupled-RoPE size, S cached tokens.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# -----------------------------------------------------------------------------
# [hybrid-attn] helpers: RoPE, sliding-window mask, KV cache

def rope_cos_sin(pos, dim, base, dtype):
    """pos: (T,) absolute positions -> cos, sin of shape (T, dim/2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=pos.device, dtype=torch.float32) / dim))
    ang = pos.to(torch.float32)[:, None] * inv_freq[None, :]
    return ang.cos().to(dtype), ang.sin().to(dtype)

def apply_rope(x, cos, sin):
    """Rotate consecutive pairs (x0,x1),(x2,x3),... of the last dim. x: (..., T, dim)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)

def sliding_window_mask(T, window, device):
    """(T, T) bool, True = may attend: causal and within the last `window` positions."""
    i = torch.arange(T, device=device)
    return (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - window)

def block_window_attention(q, k, v, W, dropout_p=0.0):
    """Sliding-window attention computing O(T * W) scores instead of O(T^2).

    Split the queries into blocks of W. A query in block b only sees keys of blocks b-1 and b,
    so every block attends to 2W keys, and all blocks but the first run as one batch.
    Same result as the mask (tested); T > W is assumed.
    """
    B, H, _, E = q.size()
    n, r = q.size(-2) // W, q.size(-2) % W                       # n full blocks, r tokens left over
    a = torch.arange(W, device=q.device)[:, None]                # query offset inside its block
    c = torch.arange(2 * W, device=q.device)[None, :]            # key offset inside blocks (b-1, b)
    mask = (c > a) & (c <= a + W)                                # (W, 2W): causal and within W

    def sdpa(q, k, v, **kw):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, **kw)

    def two_blocks(t):                                           # (B, H, nW, d) -> (B(n-1), H, 2W, d)
        return t[..., :n * W, :].unfold(2, 2 * W, W).transpose(-1, -2).transpose(1, 2).flatten(0, 1)

    out = [sdpa(q[..., :W, :], k[..., :W, :], v[..., :W, :], is_causal=True)]     # block 0: no previous block
    if n > 1:
        q_mid = q[..., W:n * W, :].view(B, H, n - 1, W, E).transpose(1, 2).flatten(0, 1)
        y = sdpa(q_mid, two_blocks(k), two_blocks(v), attn_mask=mask)             # (B(n-1), H, W, Ev)
        out.append(y.view(B, n - 1, H, W, -1).transpose(1, 2).flatten(2, 3))
    if r:                                                                         # incomplete last block
        s = (n - 1) * W
        out.append(sdpa(q[..., n * W:, :], k[..., s:, :], v[..., s:, :], attn_mask=mask[:r, :W + r]))
    return torch.cat(out, dim=-2)

def causal_attention(q, k, v, window=None, dropout_p=0.0, impl='mask'):
    """q, k: (B, H, T, E), v: (B, H, T, Ev); scale 1/sqrt(E) (= 1/sqrt(hs + dR) for MLA).

    The window has two equivalent implementations (see REPORT.md, Section 2):
    impl='mask':  naive; computes all T x T scores and loses the fused is_causal kernel.
    impl='block': optimized; computes O(T * W) scores.
    """
    T = q.size(-2)
    if window is None or window >= T:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
    if impl == 'block':
        return block_window_attention(q, k, v, window, dropout_p)
    mask = sliding_window_mask(T, window, q.device)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)

def explicit_attention_probs(q, k, window=None):
    """[hybrid-attn] Analysis only: the softmax matrix that causal_attention computes implicitly.
    q, k: (B, H, T, E) -> probs (B, H, T, T), float32. Same mask and 1/sqrt(E) scale as SDPA."""
    T = q.size(-2)
    scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(q.size(-1))
    if window is None:
        mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    else:
        mask = sliding_window_mask(T, window, q.device)
    return scores.masked_fill(~mask, float('-inf')).softmax(-1)

class KVCache:
    """Per-layer cache; every stored tensor has its time axis at dim -2.

    full    (capacity = max_len): token t is stored at slot t.
    rolling (capacity = W):       token t is stored at slot t % W, overwriting the oldest.

    Keys are cached *after* RoPE, so each one already carries its absolute position, and
    attention is invariant to the order of the keys. Hence at decode time we simply attend
    to every filled slot: they are exactly the last min(t+1, W) tokens = the window.
    """

    def __init__(self, shapes, capacity, rolling, device, dtype):
        self.capacity, self.rolling, self.length = capacity, rolling, 0
        self.bufs = [torch.zeros(*s[:-1], capacity, s[-1], device=device, dtype=dtype) for s in shapes]

    def update(self, *new):
        """Append T new tokens (one tensor (..., T, d) per buffer); return the valid cached views."""
        T, start = new[0].size(-2), self.length
        if not self.rolling:
            assert start + T <= self.capacity, "KV cache overflow: increase max_len"
            for buf, t in zip(self.bufs, new):
                buf[..., start:start + T, :] = t
        elif T == 1:
            slot = start % self.capacity
            for buf, t in zip(self.bufs, new):
                buf[..., slot:slot + 1, :] = t
        else:  # rolling prefill: only the last `capacity` tokens survive
            keep = min(T, self.capacity)
            slots = torch.arange(start + T - keep, start + T, device=new[0].device) % self.capacity
            for buf, t in zip(self.bufs, new):
                buf[..., slots, :] = t[..., T - keep:, :]
        self.length += T
        n = min(self.length, self.capacity)
        return [buf[..., :n, :] for buf in self.bufs]

    def nbytes(self):
        return sum(b.numel() * b.element_size() for b in self.bufs)

# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Upstream nanoGPT attention. [hybrid-attn] + optional RoPE, sliding window (attn='swa'), KV cache."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # [hybrid-attn] (upstream's non-flash fallback for PyTorch < 2.0 removed: SDPA is required)
        self.config = config
        self.window = config.window if config.attn == 'swa' else None
        self.cache = None

    def cache_shapes(self, B):
        hs = self.n_embd // self.n_head
        return [(B, self.n_head, hs)] * 2  # K and V, each (B, nh, S, hs)

    def forward(self, x, pos=None, use_cache=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.config.pos_emb == 'rope': # [hybrid-attn]
            cos, sin = rope_cos_sin(pos, C // self.n_head, self.config.rope_base, x.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if use_cache: # [hybrid-attn]
            k_all, v_all = self.cache.update(k, v)
        if use_cache and T == 1:
            # decode: the cache holds exactly the keys this query may see -> no mask
            y = F.scaled_dot_product_attention(q, k_all, v_all)
        else:
            # causal (optionally windowed) self-attention; (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            y = causal_attention(q, k, v, self.window, self.dropout if self.training else 0, self.config.window_impl)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    @torch.no_grad()
    def probe(self, x, pos):
        """[hybrid-attn] Analysis only (see probe.py): attention probabilities and the per-token
        quantities that would be cached. Returns dict(probs (B,H,T,T), k, v, k_nope (B,H,T,hs));
        k_nope is the key before RoPE (comparable to MLA's c_kv, which is also position-free)."""
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        k_nope = k
        if self.config.pos_emb == 'rope':
            cos, sin = rope_cos_sin(pos, C // self.n_head, self.config.rope_base, x.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        return dict(probs=explicit_attention_probs(q, k, self.window), k=k, v=v, k_nope=k_nope)

class MultiHeadLatentAttention(nn.Module):
    """[hybrid-attn] DeepSeek-V2 MLA (Sec. 2.1, eqs. 9-19); attn='mla_swa' adds the sliding window.

    Cached per token: the KV latent c_kv (dc) and the RoPE key k_R (dR) shared by all heads,
    i.e. dc + dR numbers instead of 2 * n_head * hs for MHA.
    """

    def __init__(self, config):
        super().__init__()
        C, H = config.n_embd, config.n_head
        hs, dc, dq, dR = C // H, config.kv_latent, config.q_latent, config.rope_dim
        assert dR % 2 == 0
        self.config, self.n_head, self.hs, self.dc, self.dR = config, H, hs, dc, dR
        self.window = config.window if config.attn == 'mla_swa' else None
        self.dropout = config.dropout
        # query path: low-rank only to save activation memory in training (queries are not cached)
        self.q_down = nn.Linear(C, dq, bias=False)           # W_DQ
        self.q_norm = nn.RMSNorm(dq)
        self.q_up = nn.Linear(dq, H * hs, bias=False)        # W_UQ
        self.q_rope = nn.Linear(dq, H * dR, bias=False)      # W_QR: one RoPE query per head
        # key/value path
        self.kv_down = nn.Linear(C, dc, bias=False)          # W_DKV -> c_kv (cached)
        self.kv_norm = nn.RMSNorm(dc)
        self.k_up = nn.Linear(dc, H * hs, bias=False)        # W_UK
        self.v_up = nn.Linear(dc, H * hs, bias=False)        # W_UV
        self.k_rope = nn.Linear(C, dR, bias=False)           # W_KR: ONE RoPE key for all heads (cached)
        self.c_proj = nn.Linear(H * hs, C, bias=config.bias) # W_O
        self.resid_dropout = nn.Dropout(config.dropout)
        self.cache = None

    def cache_shapes(self, B):
        # ONE buffer per token holding [c_kv | k_R]: (B, S, dc + dR). Keeping them contiguous lets
        # the absorbed decode use it directly as the single shared key (and its first dc dims as value).
        return [(B, self.dc + self.dR)]

    def forward(self, x, pos, use_cache=False):
        B, T, _ = x.size()
        H, hs, dR = self.n_head, self.hs, self.dR
        cos, sin = rope_cos_sin(pos, dR, self.config.rope_base, x.dtype)

        c_q = self.q_norm(self.q_down(x))                                     # (B, T, dq)
        q_c = self.q_up(c_q).view(B, T, H, hs).transpose(1, 2)                # (B, H, T, hs)
        q_r = self.q_rope(c_q).view(B, T, H, dR).transpose(1, 2)              # (B, H, T, dR)
        q_r = apply_rope(q_r, cos, sin)
        c_kv = self.kv_norm(self.kv_down(x))                                  # (B, T, dc)
        k_r = apply_rope(self.k_rope(x), cos, sin)                            # (B, T, dR)

        if use_cache:
            (lat_all,) = self.cache.update(torch.cat([c_kv, k_r], dim=-1))    # (B, S, dc + dR)
        if use_cache and T == 1:
            y = self._decode_absorbed(q_c, q_r, lat_all)
        else:
            y = self._attend_expanded(q_c, q_r, c_kv, k_r)
        y = y.transpose(1, 2).contiguous().view(B, T, H * hs)
        return self.resid_dropout(self.c_proj(y))

    @torch.no_grad()
    def probe(self, x, pos):
        """[hybrid-attn] Analysis only (see probe.py): attention probabilities, the cached latent
        c_kv (B,T,dc) and the re-expanded per-head k (content + RoPE part), v (B,H,T,hs)."""
        B, T, _ = x.size()
        H, hs, dR = self.n_head, self.hs, self.dR
        cos, sin = rope_cos_sin(pos, dR, self.config.rope_base, x.dtype)
        c_q = self.q_norm(self.q_down(x))
        q_c = self.q_up(c_q).view(B, T, H, hs).transpose(1, 2)
        q_r = apply_rope(self.q_rope(c_q).view(B, T, H, dR).transpose(1, 2), cos, sin)
        c_kv = self.kv_norm(self.kv_down(x))
        k_r = apply_rope(self.k_rope(x), cos, sin)
        k_c = self.k_up(c_kv).view(B, T, H, hs).transpose(1, 2)
        v = self.v_up(c_kv).view(B, T, H, hs).transpose(1, 2)
        q = torch.cat([q_c, q_r], dim=-1)
        k = torch.cat([k_c, k_r.unsqueeze(1).expand(B, H, T, dR)], dim=-1)
        return dict(probs=explicit_attention_probs(q, k, self.window), k=k, v=v, c_kv=c_kv)

    def _attend_expanded(self, q_c, q_r, c_kv, k_r):
        """Training / prefill: re-expand per-head K, V from the latent, then ordinary attention."""
        B, S, _ = c_kv.size()
        H, hs, dR = self.n_head, self.hs, self.dR
        k_c = self.k_up(c_kv).view(B, S, H, hs).transpose(1, 2)               # (B, H, S, hs)
        v = self.v_up(c_kv).view(B, S, H, hs).transpose(1, 2)                 # (B, H, S, hs)
        q = torch.cat([q_c, q_r], dim=-1)                                     # (B, H, T, hs+dR)
        k = torch.cat([k_c, k_r.unsqueeze(1).expand(B, H, S, dR)], dim=-1)    # (B, H, S, hs+dR)
        return causal_attention(q, k, v, self.window, self.dropout if self.training else 0, self.config.window_impl)

    def _decode_absorbed(self, q_c, q_r, lat):
        """Absorption trick (DeepSeek-V2 Sec. 2.1.2): never materialise per-head K/V of the cache.

            q_c . k_j     = q_c . (W_UK^h c_j) = (W_UK^h^T q_c) . c_j
            sum_j p_j v_j = W_UV^h (sum_j p_j c_j)

        After absorption every head attends to the SAME key [c_j | k_R_j] (dc + dR dims) and the
        SAME value c_j (dc dims): decode is multi-query attention with H query heads and one
        shared KV head. We run it as a single SDPA call with the H heads in the "query length" slot.
        q_c (B,H,1,hs)  q_r (B,H,1,dR)  lat (B,S,dc+dR) = [c_kv | k_R]
        """
        B, H, hs, dc = q_c.size(0), self.n_head, self.hs, self.dc
        W_UK = self.k_up.weight.view(H, hs, dc)                               # (H, hs, dc)
        W_UV = self.v_up.weight.view(H, hs, dc)                               # (H, hs, dc)
        q_lat = torch.einsum('bhtd,hdc->bhtc', q_c, W_UK)                     # (B, H, 1, dc)
        q = torch.cat([q_lat, q_r], dim=-1).view(B, 1, H, dc + self.dR)       # heads -> query slots
        k = lat.unsqueeze(1)                                                  # (B, 1, S, dc+dR)
        v = lat[..., :dc].unsqueeze(1)                                        # (B, 1, S, dc)
        o_lat = F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(hs + self.dR))  # (B,1,H,dc)
        return torch.einsum('bhc,hdc->bhd', o_lat.squeeze(1), W_UV).unsqueeze(2)  # (B, H, 1, hs)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # [hybrid-attn] choose the attention variant
        self.attn = MultiHeadLatentAttention(config) if config.attn in ('mla', 'mla_swa') else CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, pos=None, use_cache=False):
        x = x + self.attn(self.ln_1(x), pos, use_cache)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # [hybrid-attn] new fields; defaults reproduce upstream nanoGPT
    attn: str = 'mha'         # 'mha' | 'swa' | 'mla' | 'mla_swa'
    pos_emb: str = 'learned'  # 'learned' (upstream wpe table) | 'rope'
    window: int = 64          # W for swa / mla_swa (keys visible per query, including itself)
    window_impl: str = 'mask' # training / prefill: 'mask' (naive, O(T^2)) | 'block' (optimized, O(T*W))
    kv_latent: int = 64       # dc  (MLA)
    q_latent: int = 96        # dq  (MLA)
    rope_dim: int = 16        # dR  (MLA decoupled RoPE, per head)
    rope_base: float = 10000.0

    def __post_init__(self):
        assert self.attn in ('mha', 'swa', 'mla', 'mla_swa')
        assert self.pos_emb in ('learned', 'rope')
        assert self.window_impl in ('mask', 'block')
        if self.attn in ('mla', 'mla_swa'):
            assert self.pos_emb == 'rope', "MLA's position information is its decoupled RoPE"

    @property
    def windowed(self):
        return self.attn in ('swa', 'mla_swa')

    def cache_elems_per_token_per_layer(self):
        """Cached scalars per token per layer: 2*n_head*hs (MHA) or dc + dR (MLA)."""
        return self.kv_latent + self.rope_dim if self.attn in ('mla', 'mla_swa') else 2 * self.n_embd

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        if config.pos_emb == 'learned': # [hybrid-attn] no absolute table when using RoPE
            self.transformer.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and 'wpe' in self.transformer:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def get_num_attn_params(self): # [hybrid-attn]
        return sum(p.numel() for b in self.transformer.h for p in b.attn.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, start_pos=None, all_logits=False):
        """[hybrid-attn] start_pos=None: plain forward (as upstream). start_pos=int: incremental
        forward through the KV cache (call setup_cache first). all_logits: return logits for
        every position even without targets."""
        device = idx.device
        b, t = idx.size()
        use_cache = start_pos is not None
        if use_cache:
            assert start_pos == self.transformer.h[0].attn.cache.length, "KV cache out of sync"
            assert start_pos == 0 or t == 1, "multi-token steps are only supported as the initial prefill"
        p0 = start_pos or 0
        if self.config.pos_emb == 'learned':
            assert p0 + t <= self.config.block_size, f"Cannot forward sequence of length {p0 + t}, block size is only {self.config.block_size}"
        pos = torch.arange(p0, p0 + t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if self.config.pos_emb == 'learned':
            tok_emb = tok_emb + self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x, pos, use_cache)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        elif all_logits:
            logits = self.lm_head(x)
            loss = None
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    # [hybrid-attn] KV cache management ------------------------------------------------
    def setup_cache(self, batch_size, max_len, dtype=None):
        """Allocate per-layer caches. Windowed variants get a rolling buffer of W slots."""
        device = self.lm_head.weight.device
        dtype = dtype or self.lm_head.weight.dtype
        cap = min(self.config.window, max_len) if self.config.windowed else max_len
        for block in self.transformer.h:
            block.attn.cache = KVCache(block.attn.cache_shapes(batch_size), cap, self.config.windowed, device, dtype)

    def clear_cache(self):
        for block in self.transformer.h:
            block.attn.cache = None

    def cache_nbytes(self):
        return sum(block.attn.cache.nbytes() for block in self.transformer.h)

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if 'wpe' in self.transformer:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        [hybrid-attn] use_cache=True: prefill once, then decode one token at a time via the KV
        cache (requires pos_emb='rope' or a total length <= block_size).
        """
        if use_cache:
            T = idx.size(1)
            self.setup_cache(idx.size(0), T + max_new_tokens)
            logits, _ = self(idx, start_pos=0)
        for i in range(max_new_tokens):
            if not use_cache:
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                # forward the model to get the logits for the index in the sequence
                logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            if use_cache and i < max_new_tokens - 1:
                logits, _ = self(idx_next, start_pos=T + i)
        if use_cache:
            self.clear_cache()

        return idx
