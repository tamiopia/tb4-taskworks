# Quantized KV cache under a fixed memory budget

Write a streaming KV-cache codec for the small decoder-only transformer in
`/app/model`. The codec must store the model's attention keys and values in far fewer
bytes than float32 while keeping the model's next-token distribution close to
full-precision inference.

Deliver one self-contained file, `/app/submission/kvcache.py`, defining the `KVCodec`
class specified in `/app/TASK_CONTRACT.md`: `append`, `get`, `to_bytes` and
`from_bytes`. Keys arrive post-RoPE. The `i`-th token appended to a layer sits at
position `i`.

The evaluation drives your codec exactly like an inference server would. After an
exact prefill, the prompt's keys and values are appended, possibly in several chunks.
Then each teacher-forced decode step does the following in a fresh, sandboxed
process:

1. It restores the cache from the previous step's blob.
2. For each layer in turn, it appends that layer's new token and calls `get()` to
   obtain the keys and values the model attends over.
3. It serializes the cache again.

Nothing survives between steps except the blob.

Every case must satisfy all of the following:

- after every serialization, `len(blob) <= 960 * T + 65536` bytes, where `T` is the
  number of cached tokens per layer. That is 3.75 bits per key/value element plus a
  fixed 64 KiB, and it covers all data and metadata;
- the mean over decode steps of `KL(p_t || q_t)` is at most 0.012 nats, where `p_t`
  is the exact next-token distribution and `q_t` the one produced with your codec;
- the maximum of `KL(p_t || q_t)` over steps is at most 0.06 nats;
- prefill takes at most 20 s and each decode-step process at most 3 s, on one CPU
  thread.

The hidden evaluation uses weights produced by `/app/model/generate.py` with unseen
seeds, and unseen prompts and continuations. It checks four criteria:

- the interface contract;
- single-chunk prompts of 256 to 2048 tokens;
- prompts delivered in several chunks, including a one-token first chunk;
- short prompts followed by 192 to 256 decode steps.

The reward is 1 only if every criterion passes.

`/app/harness/run_eval.py` applies the same protocol, limits, sandboxing and
criteria to development cases of each kind, using `/app/model/dev_weights.npz`.
Only the Python 3.11 standard library and NumPy 2.1.3 are available, with no network
access.
