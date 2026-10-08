"""Runs one KVCodec session in its own process; driven over stdin/stdout.

Usage: python -I codec_worker.py /path/to/kvcache.py

Frames (both directions): 1-byte opcode, 8-byte little-endian payload length,
payload. Arrays are encoded as: dtype string length (1 byte), dtype string,
ndim (1 byte), shape (8 bytes each), raw C-order bytes.
"""

import importlib.util
import io
import os
import struct
import sys
import traceback


def main():
    # Keep a private handle on the protocol pipe; anything the codec prints
    # goes to stderr instead of corrupting the protocol stream.
    proto_out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    proto_in = os.fdopen(os.dup(0), "rb", buffering=0)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)

    import numpy as np

    def read_exact(n):
        buf = bytearray()
        while len(buf) < n:
            chunk = proto_in.read(n - len(buf))
            if not chunk:
                raise EOFError
            buf += chunk
        return bytes(buf)

    def send(op, payload=b""):
        proto_out.write(op + struct.pack("<Q", len(payload)) + payload)

    def dec_array(b):
        bio = io.BytesIO(b)
        (n,) = struct.unpack("<B", bio.read(1))
        dt = np.dtype(bio.read(n).decode())
        (nd,) = struct.unpack("<B", bio.read(1))
        shape = struct.unpack("<" + "Q" * nd, bio.read(8 * nd))
        return np.frombuffer(bio.read(), dtype=dt).reshape(shape)

    def enc_array(a):
        a = np.asarray(a)
        if a.dtype.hasobject or a.dtype.kind not in "biufc":
            raise TypeError(f"get() returned an array of unsupported dtype {a.dtype}")
        a = np.ascontiguousarray(a)
        ds = a.dtype.str.encode()
        return (struct.pack("<B", len(ds)) + ds + struct.pack("<B", a.ndim)
                + struct.pack("<" + "Q" * a.ndim, *a.shape) + a.tobytes())

    try:
        spec = importlib.util.spec_from_file_location("kvcache", sys.argv[1])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["kvcache"] = mod
        spec.loader.exec_module(mod)
        Codec = mod.KVCodec
    except BaseException:
        send(b"E", ("failed to import KVCodec:\n" + traceback.format_exc()).encode())
        return
    send(b"R")

    codec = None
    while True:
        try:
            op = read_exact(1)
        except EOFError:
            return
        (n,) = struct.unpack("<Q", read_exact(8))
        payload = read_exact(n)
        try:
            if op == b"I":
                L, H, D = struct.unpack("<III", payload)
                codec = Codec(L, H, D)
                send(b"O")
            elif op == b"F":
                L, H, D = struct.unpack("<III", payload[:12])
                codec = Codec.from_bytes(payload[12:], L, H, D)
                send(b"O")
            elif op == b"A":
                (layer, nk) = struct.unpack("<IQ", payload[:12])
                k = dec_array(payload[12:12 + nk]).copy()
                v = dec_array(payload[12 + nk:]).copy()
                codec.append(layer, k, v)
                send(b"O")
            elif op == b"G":
                (layer,) = struct.unpack("<I", payload)
                res = codec.get(layer)
                if not isinstance(res, tuple) or len(res) != 2:
                    raise TypeError("get() must return a tuple (K_hat, V_hat)")
                ka, va = enc_array(res[0]), enc_array(res[1])
                send(b"O", struct.pack("<Q", len(ka)) + ka + va)
            elif op == b"B":
                blob = codec.to_bytes()
                if not isinstance(blob, (bytes, bytearray)):
                    raise TypeError("to_bytes() must return bytes")
                send(b"O", bytes(blob))
            elif op == b"Q":
                send(b"O")
                return
            else:
                raise ValueError(f"unknown opcode {op!r}")
        except BaseException:
            send(b"E", traceback.format_exc().encode())
            return


if __name__ == "__main__":
    main()
