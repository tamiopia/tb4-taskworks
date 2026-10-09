# KV-cache codec contract

## Model

`/app/model/model.py` is the float32 NumPy forward pass used by the evaluation:
4 layers, 4 heads per layer, `head_dim = 64`, vocabulary 512, pre-RMSNorm blocks,
rotary position embeddings in the rotate-half convention with base 10000.
`/app/model/generate.py` produces weights from a seed; `/app/model/dev_weights.npz`
was produced from the development seed. Position 0 of every sequence is the BOS token.

## Interface

`/app/submission/kvcache.py` must be a single self-contained file (only this file is
loaded) defining:

```python
class KVCodec:
    def __init__(self, n_layers: int, n_heads: int, head_dim: int) -> None: ...
    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> None: ...
    def get(self, layer: int) -> tuple[np.ndarray, np.ndarray]: ...
    def to_bytes(self) -> bytes: ...
    @classmethod
    def from_bytes(cls, blob: bytes, n_layers: int, n_heads: int, head_dim: int) -> "KVCodec": ...
```

- `append` receives float32 arrays of shape `[n_heads, t, head_dim]`, `t >= 1`. Keys
  are post-RoPE. The `i`-th token appended to a layer (counting from 0) is at
  position `i` and its key was rotated for position `i`.
- `get` returns `(K_hat, V_hat)` for every token appended to that layer, in order:
  two finite `float32` arrays of shape `[n_heads, T, head_dim]`.
- `to_bytes` serializes the complete state; `from_bytes` rebuilds it.

Only the Python 3.11 standard library and NumPy 2.1.3 are available. There is no
network access.

## Protocol for one case

A case is a prompt of `P` tokens and a teacher-forced continuation of `D` tokens.
The harness runs it once with an exact float32 cache and once with your codec.

1. Prefill: the prompt goes through the model with exact attention. The harness calls
   `KVCodec(4, 4, 64)`, then `append(layer, k, v)` for every layer, possibly in several
   consecutive chunks of any size (a chunk may hold a single token), then `to_bytes()`.
2. Decode step, repeated `D` times: `from_bytes(previous_blob, 4, 4, 64)`; then, for each
   layer in order, `append` of the new token's `k, v` followed by `get`, whose result the
   harness uses for that layer's attention; finally `to_bytes()`.

Prefill and every decode step run in a new Python process under a fresh unprivileged
user id that cannot read the evaluation data. When a step ends, every process it
started is killed, every file it created is deleted, and SysV IPC objects are
removed. Only the blob carries state between steps.

## Requirements (every case)

- Budget: after every `to_bytes()`, `len(blob) <= 960 * T + 65536` bytes, where `T`
  is the number of tokens cached per layer (3.75 bits per key/value element plus
  64 KiB, covering all data and metadata).
- Accuracy: with `p_t` the exact and `q_t` the codec's next-token distribution at
  decode step `t`, mean over steps of `KL(p_t || q_t)` at most 0.012 nats and the
  maximum over steps at most 0.06 nats.
- Time: 20 s for the prefill process and 3 s for each decode-step process, including
  interpreter start-up and import, on one CPU thread; the harness's own model
  computation is not counted.

## Hidden evaluation

Weights come from `generate.py` with seeds other than the development seed, and
prompts and continuations are sampled from those models. Cases are grouped into
criteria; a criterion passes only if all of its cases pass:

| Criterion | Cases |
|---|---|
| interface | contract checks on synthetic data: shapes, dtype, chunked appends, cross-process restore |
| prefill_accuracy | single-chunk prompts of 256 to 2048 tokens, `D = 64` |
| chunked_prefill | prompts of about 1000 to 1800 tokens delivered in several chunks, including a first chunk of one token |
| decode_streaming | short prompts (about 100 to 160 tokens) followed by `D = 192` to `256` decode steps |

The reward is 1 only when every criterion passes. `/app/harness/run_eval.py` runs the
same protocol, limits, isolation and grouping on development cases of each kind.
