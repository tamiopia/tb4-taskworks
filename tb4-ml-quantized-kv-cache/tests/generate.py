"""Synthetic weight generator for the task model.

    python generate.py --seed 1234 --out weights.npz

The weights are random but structured to reproduce the KV-cache statistics of
real decoder-only LLMs:

* a residual "massive activation" channel that is (nearly) constant for every
  token and persists through all layers;
* a handful of key channels per head, in low-frequency RoPE dimensions, that
  read that massive channel and are therefore 10-30x larger than the rest of
  the key vector for every token (key outlier channels);
* an attention sink: the BOS token writes a dedicated residual channel that
  every head's keys pick up in a low-frequency RoPE pair, while every query
  carries a constant component on the same pair, so a large share of
  attention mass goes to position 0;
* recency heads: constant query/key components on the highest-frequency RoPE
  pairs, so attention concentrates on nearby positions -- and those key
  channels oscillate with position after RoPE;
* heads with different content-attention temperatures.
"""

import argparse

import numpy as np

from model import D_FF, D_MODEL, HEAD_DIM, N_HEADS, N_LAYERS, VOCAB

MASSIVE = 0  # residual channel that is constant for every token
SINK = 1  # residual channel set only by the BOS token
# Shape parameters of the generator (shared by the development and hidden weights).
P = {
    "temps": [0.2, 0.3, 0.4, 0.5],  # per-head content attention temperatures
    "unemb": 2.0,  # output projection scale
    "wo": 0.7,  # attention output projection scale
    "n_local": 3,  # recency heads per layer
    "local_pairs": 4,  # high-frequency RoPE pairs used by recency heads
    "local_k": 3.5,
    "local_q": 0.4,
}


def generate(seed):
    rng = np.random.default_rng(seed)
    half = HEAD_DIM // 2
    d = D_MODEL

    def normal(*shape, std=1.0):
        return (rng.standard_normal(shape) * std).astype(np.float32)

    w = {}
    emb = normal(VOCAB, d)
    emb[:, MASSIVE] = rng.uniform(5.0, 6.0)
    emb[:, SINK] = 0.0
    emb[0, :] = normal(d, std=0.3)  # BOS: mostly the sink channel
    emb[0, MASSIVE] = emb[1, MASSIVE]
    emb[0, SINK] = 14.0
    w["emb"] = emb

    for layer in range(N_LAYERS):
        temps = rng.permutation(np.array(P["temps"], dtype=np.float32))
        wq = np.zeros((d, d), np.float32)
        local_heads = rng.choice(N_HEADS, size=P["n_local"], replace=False)
        wk = np.zeros((d, d), np.float32)
        for h in range(N_HEADS):
            cols = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            sq = np.sqrt(temps[h])
            wq[:, cols] = normal(d, HEAD_DIM, std=sq / np.sqrt(d))
            wk[:, cols] = normal(d, HEAD_DIM, std=sq / np.sqrt(d))
            # Special rows contribute only through the channels set below.
            wq[[MASSIVE, SINK], cols] = 0.0
            wk[[MASSIVE, SINK], cols] = 0.0

            # Low-frequency RoPE pairs: index i pairs channel i with i + half.
            pairs = rng.choice(np.arange(20, half - 1), size=3, replace=False)
            sink_pair = half - 1
            base = h * HEAD_DIM
            for i in pairs:  # key outlier channels
                for c in (i, i + half):
                    wk[MASSIVE, base + c] = rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 4.0)
            if h in local_heads:  # recency: constant q/k on high-frequency pairs
                for i in range(P["local_pairs"]):
                    wk[MASSIVE, base + i] = P["local_k"] * rng.uniform(0.8, 1.2)
                    wq[MASSIVE, base + i] = P["local_q"] * rng.uniform(0.8, 1.2)
            for c in (sink_pair, sink_pair + half):  # attention sink
                wk[SINK, base + c] = rng.uniform(0.6, 0.9)
                wq[MASSIVE, base + c] = rng.uniform(0.45, 0.6)
        w[f"wq{layer}"] = wq
        w[f"wk{layer}"] = wk
        wv = normal(d, d, std=1.0 / np.sqrt(d))
        wv[[MASSIVE, SINK], :] = 0.0
        w[f"wv{layer}"] = wv
        wo = normal(d, d, std=P["wo"] / np.sqrt(d))
        wo[:, [MASSIVE, SINK]] = 0.0
        w[f"wo{layer}"] = wo
        w1 = normal(d, D_FF, std=1.0 / np.sqrt(d))
        w2 = normal(D_FF, d, std=0.5 / np.sqrt(D_FF))
        w2[:, [MASSIVE, SINK]] = 0.0
        w[f"w1_{layer}"] = w1
        w[f"w2_{layer}"] = w2
        w[f"g1_{layer}"] = np.ones(d, np.float32)
        w[f"g2_{layer}"] = np.ones(d, np.float32)

    w["gf"] = np.ones(d, np.float32)
    unemb = normal(d, VOCAB, std=1.0 / np.sqrt(d))
    unemb[[MASSIVE, SINK], :] = 0.0
    w["unemb"] = unemb * np.float32(P["unemb"])
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    np.savez_compressed(a.out, **generate(a.seed))


if __name__ == "__main__":
    main()
