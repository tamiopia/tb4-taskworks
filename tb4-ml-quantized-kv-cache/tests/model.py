"""Reference float32 forward pass for the task's decoder-only transformer.

Architecture: pre-RMSNorm residual blocks, multi-head causal self-attention with
rotary position embeddings (half-split / "rotate_half" convention), GELU MLP,
untied output projection. Everything is float32 NumPy.
"""

import numpy as np

N_LAYERS = 4
N_HEADS = 4
HEAD_DIM = 64
D_MODEL = N_HEADS * HEAD_DIM
D_FF = 4 * D_MODEL
VOCAB = 512
ROPE_BASE = 10000.0
BOS = 0
EPS = 1e-6


def rms_norm(x, g):
    x = x.astype(np.float32, copy=False)
    r = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)
    return (x / r) * g


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.float32(0.7978845608) * (x + np.float32(0.044715) * x * x * x)))


def rope_angles(positions):
    half = HEAD_DIM // 2
    inv_freq = ROPE_BASE ** (-np.arange(half, dtype=np.float64) * 2.0 / HEAD_DIM)
    ang = np.asarray(positions, dtype=np.float64)[:, None] * inv_freq[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def apply_rope(x, cos, sin):
    """x: [H, T, HEAD_DIM]; cos, sin: [T, HEAD_DIM // 2]."""
    half = HEAD_DIM // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x):
    x = x.astype(np.float64)
    m = np.max(x)
    return x - m - np.log(np.sum(np.exp(x - m)))


class Model:
    def __init__(self, weights):
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls({k: f[k] for k in f.files})

    def _qkv(self, layer, h, positions):
        w = self.w
        t = h.shape[0]
        q = (h @ w[f"wq{layer}"]).reshape(t, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
        k = (h @ w[f"wk{layer}"]).reshape(t, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
        v = (h @ w[f"wv{layer}"]).reshape(t, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
        cos, sin = rope_angles(positions)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        return (np.ascontiguousarray(q, dtype=np.float32),
                np.ascontiguousarray(k, dtype=np.float32),
                np.ascontiguousarray(v, dtype=np.float32))

    def _mlp(self, layer, x):
        w = self.w
        h = rms_norm(x, w[f"g2_{layer}"])
        return x + gelu(h @ w[f"w1_{layer}"]) @ w[f"w2_{layer}"]

    def _attn_out(self, layer, o):
        """o: [H, t, HEAD_DIM] -> [t, D_MODEL] projected."""
        t = o.shape[1]
        return o.transpose(1, 0, 2).reshape(t, D_MODEL) @ self.w[f"wo{layer}"]

    def prefill(self, tokens):
        """Exact causal forward over `tokens` at positions 0..P-1.

        Returns (kv, logits_last): kv is a list with one (k, v) pair per layer,
        each float32 of shape [N_HEADS, P, HEAD_DIM] (keys post-RoPE)."""
        w = self.w
        tokens = np.asarray(tokens, dtype=np.int64)
        p = tokens.shape[0]
        x = w["emb"][tokens]
        mask = np.triu(np.full((p, p), -np.inf, dtype=np.float32), k=1)
        scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
        kv = []
        for layer in range(N_LAYERS):
            h = rms_norm(x, w[f"g1_{layer}"])
            q, k, v = self._qkv(layer, h, np.arange(p))
            kv.append((k, v))
            s = np.matmul(q, k.transpose(0, 2, 1)) * scale + mask
            o = np.matmul(softmax(s), v)
            x = x + self._attn_out(layer, o)
            x = self._mlp(layer, x)
        logits = rms_norm(x[-1:], w["gf"]) @ w["unemb"]
        return kv, logits[0]

    def decode_step(self, token, pos, kv_provider):
        """Run one token at position `pos`.

        kv_provider(layer, k_new, v_new) receives this token's key/value
        ([N_HEADS, 1, HEAD_DIM] each), must store them, and returns (K, V) for
        every cached token of that layer including this one ([N_HEADS, T, HEAD_DIM]).
        Returns float32 next-token logits of shape [VOCAB]."""
        w = self.w
        x = w["emb"][np.array([token], dtype=np.int64)]
        scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
        for layer in range(N_LAYERS):
            h = rms_norm(x, w[f"g1_{layer}"])
            q, k, v = self._qkv(layer, h, np.array([pos]))
            K, V = kv_provider(layer, k, v)
            s = np.matmul(q, K.transpose(0, 2, 1)) * scale
            o = np.matmul(softmax(s), V)
            x = x + self._attn_out(layer, o)
            x = self._mlp(layer, x)
        return (rms_norm(x, w["gf"]) @ w["unemb"])[0]


class ExactCache:
    """Full-precision float32 cache, usable as a decode_step kv_provider."""

    def __init__(self, kv=None):
        self.kc = [None] * N_LAYERS
        self.vc = [None] * N_LAYERS
        if kv is not None:
            for layer, (k, v) in enumerate(kv):
                self.kc[layer], self.vc[layer] = k, v

    def __call__(self, layer, k, v):
        if self.kc[layer] is None:
            self.kc[layer], self.vc[layer] = k, v
        else:
            self.kc[layer] = np.concatenate([self.kc[layer], k], axis=1)
            self.vc[layer] = np.concatenate([self.vc[layer], v], axis=1)
        return self.kc[layer], self.vc[layer]
