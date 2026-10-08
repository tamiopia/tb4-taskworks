# Quantized KV-cache codec under a fixed memory budget

`/app` contains a small decoder-only transformer and an evaluation harness. Your job
is to write a KV-cache codec in `/app/kvcache.py`. The codec has to store the
model's attention keys and values in far fewer bytes than float32 while keeping the
model's next-token distribution close to full-precision inference.

## What is in `/app`

- `/app/model/model.py`: the reference forward pass (NumPy, float32). It has
  4 layers, 4 attention heads per layer, `head_dim = 64`, rotary position embeddings
  (RoPE, "rotate-half" convention, base 10000), and a vocabulary of 512 tokens.
- `/app/model/generate.py`: the generator that produces the model weights from a seed.
- `/app/model/dev_weights.npz`: development weights, from `generate.py`.
- `/app/data/dev_cases.json`: development prompts and continuations.
- `/app/harness/run_eval.py`: runs the same protocol and checks as the hidden
  evaluation, but on the development weights and cases. Run it with
  `python /app/harness/run_eval.py`. Add `--no-isolate` for faster runs without the
  per-step sandbox.

You may read and run all of these files. The hidden evaluation uses its own copies,
so changing them has no effect on grading.

## What you must write

Write `/app/kvcache.py`. It must be a single self-contained file, because only this
file is loaded. It must define a class `KVCodec` with this interface:

```python
import numpy as np

class KVCodec:
    def __init__(self, n_layers: int, n_heads: int, head_dim: int) -> None:
        """Create an empty cache."""

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        """Append t >= 1 new tokens to `layer`.
        k, v: float32, shape [n_heads, t, head_dim]. Keys are post-RoPE."""

    def get(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (K_hat, V_hat) for every token cached in `layer`, in insertion order.
        Both are float32 arrays of shape [n_heads, T, head_dim]."""

    def to_bytes(self) -> bytes:
        """Serialize the complete cache state."""

    @classmethod
    def from_bytes(cls, blob: bytes, n_layers: int, n_heads: int,
                   head_dim: int) -> "KVCodec":
        """Rebuild a cache from the output of to_bytes()."""
```

Tokens are appended in order, starting from position 0. The `i`-th token appended to
a layer (counting from 0) sits at position `i`, and its key was rotated by RoPE for
position `i`. Position 0 always holds the BOS token.

Only the Python 3.11 standard library and NumPy 2.1.3 (already installed) are
available. The evaluation has no network access. Do not modify the Python
installation; the evaluation checks that it is unchanged.

## Evaluation protocol

Each evaluation case consists of a prompt of `P` tokens followed by a fixed
continuation of `D = 64` tokens. The continuation is teacher-forced: every step feeds
the next token of the continuation, whatever your codec does. The harness runs the
case twice, once with an exact float32 cache and once with your codec, and compares
the two runs.

1. **Prefill.** The harness runs the prompt through the model with exact float32
   attention. It creates `KVCodec(4, 4, 64)` and calls `append(layer, k, v)` for
   every layer. The prompt may arrive in one chunk or in several consecutive chunks.
   It then calls `to_bytes()`.
2. **Decode step** (repeated `D` times). The harness calls `from_bytes(previous_blob, 4, 4, 64)`.
   For each layer in order, it computes the new token's `k` and `v` from the
   hidden state produced by the previous layer, calls `append(layer, k, v)`, then calls
   `get(layer)`. It computes attention over the returned `K_hat` and `V_hat` with its
   own code. After the last layer it records the next-token logits and calls
   `to_bytes()`.

Prefill and every decode step run your codec in a new Python process, under a fresh
unprivileged user id. That process cannot read the model weights, the token ids or
the evaluation files. Its only input from earlier steps is the previous blob. When a
step ends, every process the codec started is killed. Every file it created is
deleted, along with any shared-memory objects. Any state that is not in the blob is
lost.

## Requirements

All of the following must hold for every case.

**Memory budget.** After every call to `to_bytes()` (prefill and each decode step),
with `T` tokens cached per layer:

```
len(blob) <= 960 * T + 65536        # bytes
```

That allowance is 3.75 bits per cached key/value element, plus a fixed 64 KiB. It
covers everything: quantized data, scales, zero points, outliers, metadata, and any
tokens you keep at higher precision. For comparison, float16 uses 16 bits per element.

**Accuracy.** Let `p_t` be the full-precision next-token distribution at decode step
`t`, and `q_t` the distribution from the run with your codec.

- Mean of `KL(p_t || q_t)` over the 64 decode steps: at most **0.012** nats.
- Maximum of `KL(p_t || q_t)` over any single step: at most **0.06** nats.

**Interface and resources.**

- `get()` returns a tuple of two finite `float32` arrays of shape `[4, T, 64]`, where `T` is
  the number of tokens appended to that layer so far.
- The prefill process (start-up, import, all appends, `to_bytes`) has 20 seconds.
  Each decode-step process (start-up, import, `from_bytes`, all appends and gets,
  `to_bytes`) has 5 seconds. Time the harness spends on its own model computation
  does not count. Your code runs on one CPU thread (`OPENBLAS_NUM_THREADS=1`).

## Hidden evaluation

The hidden evaluation uses weights produced by `/app/model/generate.py` with seeds
other than the development seed. It also uses different prompts, sampled from those
models. Prompt lengths `P` range from 256 to 2048 tokens. Your codec never receives
the weights, so it cannot use statistics measured from the development weights. It
has to adapt to the keys and values it is given.

`run_eval.py` applies the same thresholds, time limits and process isolation as the
hidden evaluation. Passing it on the development cases is necessary but does not
guarantee passing the hidden evaluation.
