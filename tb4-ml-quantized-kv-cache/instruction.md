# Quantized KV-cache codec under a fixed memory budget

`/app` contains a small decoder-only transformer and an evaluation harness. Your job
is to write a KV-cache codec in `/app/kvcache.py`. The codec has to store the
model's attention keys and values in far fewer bytes than float32 while keeping the
model's next-token distribution close to full-precision inference.

## What is in `/app`

- `/app/model/model.py`: the reference forward pass (NumPy, float32). It has
  4 layers, 4 attention heads per layer, `head_dim = 64`, rotary position embeddings
  (RoPE), and a vocabulary of 512 tokens.
- `/app/model/dev_weights.npz`: development weights.
- `/app/data/dev_cases.json`: development prompts and continuations.
- `/app/harness/run_eval.py`: runs the same protocol and checks as the hidden
  evaluation, but on the development weights and cases. Run it with
  `python /app/harness/run_eval.py`.

You may read and run all of these files. Do not modify them. The hidden evaluation
uses its own copies.

## What you must write

Write `/app/kvcache.py`. It must define a class `KVCodec` with this interface:

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

Only the Python 3.11 standard library and NumPy (already installed) are available.
The evaluation has no network access.

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
2. **Decode step** (repeated `D` times). The harness calls `from_bytes(previous_blob, ...)`.
   For each layer in order, it computes the new token's `k` and `v` from the
   hidden state produced by the previous layer, calls `append(layer, k, v)`, then calls
   `get(layer)`. It computes attention over the returned `K_hat` and `V_hat` with its
   own code. After the last layer it records the next-token logits and calls
   `to_bytes()`.

Every decode step runs your codec in a new Python process, as an unprivileged user.
That process cannot read the model weights or the token ids, and its only input from
earlier steps is the previous blob. Every directory the process can write to is
emptied between steps. Any state that is not in the blob is lost.

## Requirements

All of the following must hold for every case.

**Memory budget.** After every call to `to_bytes()` (prefill and each decode step),
with `T` tokens cached:

```
len(blob) <= 768 * T + 16384        # bytes
```

That allowance is 3.0 bits per cached key/value element, plus a fixed 16 KiB. It
covers everything: quantized data, scales, zero points, outliers, metadata, and any
tokens you keep at higher precision. For comparison, float16 uses 16 bits per element.

**Accuracy.** Let `p_t` be the full-precision next-token distribution at decode step
`t`, and `q_t` the distribution from the run with your codec.

- Mean of `KL(p_t || q_t)` over the 64 decode steps: at most **0.02** nats.
- Maximum of `KL(p_t || q_t)` over any single step: at most **0.10** nats.
- `argmax p_t == argmax q_t` on at least **60 of the 64** steps.

**Interface and resources.**

- `get()` returns finite float32 arrays of the documented shape.
- `from_bytes(to_bytes())` round-trips. `get()` returns bitwise-identical arrays
  before and after the round trip.
- Each decode-step process (`from_bytes`, all appends and gets, `to_bytes`) finishes
  within 5 seconds on one CPU thread. Prefill has a 20-second limit.

## Hidden evaluation

The hidden evaluation uses model weights drawn from the same generator as
`dev_weights.npz` but with a different seed. It also uses different prompts. Prompt
lengths `P` range from 256 to 2048 tokens. Your codec never receives the weights, so it
cannot use statistics measured from the development weights. It has to adapt to
the keys and values it is given.

`run_eval.py` applies the exact thresholds and process isolation listed above. Passing
it on the development cases is necessary but does not guarantee passing the hidden
evaluation.
