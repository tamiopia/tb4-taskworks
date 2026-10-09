"""Evaluation protocol for KVCodec implementations.

The same file is used by /app/harness/run_eval.py (development cases) and by
the hidden verifier. /app/TASK_CONTRACT.md describes the rules it enforces.
"""

import hashlib
import json
import os
import pwd
import resource
import secrets
import select
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(HERE, "model.py")):  # /app layout: harness/ next to model/
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))
sys.path.insert(0, HERE)

from model import (D_MODEL, HEAD_DIM, N_HEADS, N_LAYERS, ExactCache, Model,  # noqa: E402
                   log_softmax)

BUDGET_PER_TOKEN = 960  # bytes per cached token = 3.75 bits per K/V element
BUDGET_FIXED = 65536
MEAN_KL_MAX = 0.012
MAX_KL_MAX = 0.060
PREFILL_SECONDS = 20.0
DECODE_STEP_SECONDS = 3.0
MAX_BLOB_BYTES = 64 << 20

WORKER = os.path.join(HERE, "codec_worker.py")


def budget(n_tokens):
    return BUDGET_PER_TOKEN * n_tokens + BUDGET_FIXED


def weights_digest(weights):
    h = hashlib.sha256()
    for k in sorted(weights):
        a = np.ascontiguousarray(weights[k], dtype=np.float32)
        h.update(k.encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


class CodecError(Exception):
    pass


# --------------------------------------------------------------------------
# Wire format


def _enc_array(a):
    a = np.ascontiguousarray(a)
    ds = a.dtype.str.encode()
    return (struct.pack("<B", len(ds)) + ds + struct.pack("<B", a.ndim)
            + struct.pack("<" + "Q" * a.ndim, *a.shape) + a.tobytes())


def _dec_array(b):
    off = 0
    (n,) = struct.unpack_from("<B", b, off)
    off += 1
    dt = np.dtype(b[off:off + n].decode())
    off += n
    (nd,) = struct.unpack_from("<B", b, off)
    off += 1
    shape = struct.unpack_from("<" + "Q" * nd, b, off)
    off += 8 * nd
    count = int(np.prod(shape)) if nd else 1
    if len(b) - off != count * dt.itemsize:
        raise CodecError("malformed array from codec")
    return np.frombuffer(b, dtype=dt, offset=off, count=count).reshape(shape)


# --------------------------------------------------------------------------
# Sandbox


def _world_writable_dirs():
    """Make every regular file read-only for 'other' users and return the
    top-most directories an arbitrary uid can create entries in."""
    found = []
    for root, dirs, files in os.walk("/", topdown=True):
        if root in ("/proc", "/sys") or root.startswith(("/proc/", "/sys/")):
            dirs[:] = []
            continue
        for name in files:
            p = os.path.join(root, name)
            try:
                st = os.lstat(p)
                if stat.S_ISREG(st.st_mode) and st.st_mode & stat.S_IWOTH:
                    os.chmod(p, stat.S_IMODE(st.st_mode) & ~stat.S_IWOTH)
            except OSError:
                continue
        try:
            st = os.lstat(root)
        except OSError:
            continue
        if st.st_mode & stat.S_IWOTH:
            found.append(root)
    top = []
    for d in sorted(found):
        if not any(d.startswith(t.rstrip("/") + "/") for t in top):
            top.append(d)
    return top


def _pids_by_uid():
    res = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/status") as f:
                for line in f:
                    if line.startswith("Uid:"):
                        res[int(d)] = int(line.split()[1])
                        break
        except OSError:
            pass
    return res


def _remove_sysv_ipc():
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    IPC_RMID = 0
    for kind, fn in (("shm", "shmctl"), ("msg", "msgctl"), ("sem", "semctl")):
        try:
            with open(f"/proc/sysvipc/{kind}") as f:
                lines = f.read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            ident = int(line.split()[1])
            if kind == "sem":
                getattr(libc, fn)(ident, 0, IPC_RMID)
            else:
                getattr(libc, fn)(ident, IPC_RMID, None)


class Sandbox:
    """Runs each codec session in a fresh process.

    With isolate=True (requires root) every session runs under a fresh,
    random, unprivileged uid, cannot read the evaluation data, and everything
    it leaves behind (processes, files, SysV IPC objects) is destroyed when the
    session ends.
    """

    def __init__(self, codec_path, isolate=True, protect=()):
        if not os.path.isfile(codec_path):
            raise CodecError(f"{codec_path} does not exist")
        self.isolate = isolate
        if isolate and os.geteuid() != 0:
            raise RuntimeError("isolation requires root; use --no-isolate")
        base = tempfile.mkdtemp(prefix="kvcodec-")
        os.chmod(base, 0o755)
        self.base = base
        self.codec = os.path.join(base, "kvcache.py")
        self.worker = os.path.join(base, "codec_worker.py")
        shutil.copyfile(codec_path, self.codec)
        shutil.copyfile(WORKER, self.worker)
        os.chmod(self.codec, 0o644)
        os.chmod(self.worker, 0o644)
        self.homes = os.path.join(base, "home")
        os.mkdir(self.homes)
        os.chmod(self.homes, 0o711)
        self.logs = tempfile.mkdtemp(prefix="kvcodec-log-")
        os.chmod(self.logs, 0o700)
        if isolate:
            for p in protect:
                os.chmod(p, 0o700)
            self.ww_dirs = _world_writable_dirs()
        self._used_uids = set(pwd_uid for pwd_uid in (e.pw_uid for e in pwd.getpwall()))

    def close(self):
        shutil.rmtree(self.base, ignore_errors=True)
        shutil.rmtree(self.logs, ignore_errors=True)

    def _fresh_uid(self):
        while True:
            uid = 200000 + secrets.randbelow(1 << 30)
            if uid not in self._used_uids:
                self._used_uids.add(uid)
                return uid

    def spawn(self, seconds):
        uid = self._fresh_uid() if self.isolate else os.getuid()
        home = os.path.join(self.homes, str(uid))
        os.mkdir(home, 0o700)
        if self.isolate:
            os.chown(home, uid, uid)
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": home,
            "TMPDIR": home,
            "LANG": "C.UTF-8",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        err = open(os.path.join(self.logs, f"{uid}.err"), "w+b")

        def limits():
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
            resource.setrlimit(resource.RLIMIT_FSIZE, (256 << 20, 256 << 20))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        kw = dict(user=uid, group=uid, extra_groups=[]) if self.isolate else {}
        proc = subprocess.Popen(
            [sys.executable, "-I", self.worker, self.codec],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
            cwd=home, env=env, preexec_fn=limits if self.isolate else None,
            start_new_session=True, close_fds=True, **kw)
        return CodecProcess(self, proc, uid, home, err, seconds)

    def cleanup(self, uid, home):
        if self.isolate:
            for _ in range(50):
                pids = [p for p, u in _pids_by_uid().items() if u == uid]
                if not pids:
                    break
                for p in pids:
                    try:
                        os.kill(p, signal.SIGKILL)
                    except OSError:
                        pass
                time.sleep(0.02)
            for d in self.ww_dirs:
                _remove_owned(d, uid)
            _remove_sysv_ipc()
        shutil.rmtree(home, ignore_errors=True)


def _remove_owned(top, uid):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files + dirs:
            p = os.path.join(root, name)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if st.st_uid == uid:
                if stat.S_ISDIR(st.st_mode):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass


class CodecProcess:
    def __init__(self, sandbox, proc, uid, home, err, seconds):
        self.sb, self.proc, self.uid, self.home, self.err = sandbox, proc, uid, home, err
        self.remaining = seconds
        self.seconds = seconds
        self.closed = False
        os.set_blocking(proc.stdout.fileno(), False)
        os.set_blocking(proc.stdin.fileno(), False)
        try:
            op, payload = self._recv()
        except CodecError:
            self.close()
            raise
        if op != b"R":
            msg = payload.decode(errors="replace")
            self.close()
            raise CodecError(msg)

    # -- low level ---------------------------------------------------------
    def _wait(self, fd, write):
        t0 = time.monotonic()
        r, w, _ = select.select([] if write else [fd], [fd] if write else [], [], max(self.remaining, 0))
        self.remaining -= time.monotonic() - t0
        if not (r or w) or self.remaining <= 0:
            raise CodecError(f"time limit of {self.seconds:.0f} s exceeded")

    def _send(self, op, payload=b""):
        data = memoryview(op + struct.pack("<Q", len(payload)) + payload)
        fd = self.proc.stdin.fileno()
        while len(data):
            self._wait(fd, True)
            try:
                n = os.write(fd, data)
            except BrokenPipeError:
                raise CodecError("codec process exited unexpectedly" + self._stderr())
            data = data[n:]

    def _read(self, n):
        fd = self.proc.stdout.fileno()
        buf = bytearray()
        while len(buf) < n:
            self._wait(fd, False)
            chunk = os.read(fd, min(n - len(buf), 1 << 22))
            if not chunk:
                raise CodecError("codec process exited unexpectedly" + self._stderr())
            buf += chunk
        return bytes(buf)

    def _recv(self):
        op = self._read(1)
        (n,) = struct.unpack("<Q", self._read(8))
        if n > MAX_BLOB_BYTES * 4:
            raise CodecError("codec sent an oversized message")
        return op, self._read(n)

    def _stderr(self, limit=4000):
        try:
            self.err.flush()
            self.err.seek(0)
            txt = self.err.read().decode(errors="replace")
        except OSError:
            return ""
        return ("\n--- codec stderr ---\n" + txt[-limit:]) if txt.strip() else ""

    def _call(self, op, payload=b""):
        self._send(op, payload)
        rop, data = self._recv()
        if rop == b"E":
            raise CodecError(data.decode(errors="replace")[-4000:])
        if rop != b"O":
            raise CodecError("protocol error")
        return data

    # -- API -----------------------------------------------------------------
    def init(self, L, H, D):
        self._call(b"I", struct.pack("<III", L, H, D))

    def from_bytes(self, blob, L, H, D):
        self._call(b"F", struct.pack("<III", L, H, D) + blob)

    def append(self, layer, k, v):
        ka = _enc_array(np.asarray(k, np.float32))
        va = _enc_array(np.asarray(v, np.float32))
        self._call(b"A", struct.pack("<IQ", layer, len(ka)) + ka + va)

    def get(self, layer):
        data = self._call(b"G", struct.pack("<I", layer))
        (nk,) = struct.unpack_from("<Q", data, 0)
        return _dec_array(data[8:8 + nk]), _dec_array(data[8 + nk:])

    def to_bytes(self):
        return self._call(b"B")

    def used_seconds(self):
        return self.seconds - self.remaining

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.proc.poll() is None:
                try:
                    self.remaining = max(self.remaining, 1.0)
                    self._call(b"Q")
                except CodecError:
                    pass
        finally:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except OSError:
                pass
            self.proc.wait()
            for f in (self.proc.stdin, self.proc.stdout):
                try:
                    f.close()
                except OSError:
                    pass
            self.err.close()
            self.sb.cleanup(self.uid, self.home)


# --------------------------------------------------------------------------
# Protocol


def _check_kv(K, V, T):
    want = (N_HEADS, T, HEAD_DIM)
    for name, a in (("K_hat", K), ("V_hat", V)):
        if a.dtype != np.float32:
            raise CodecError(f"get() returned {name} with dtype {a.dtype}, expected float32")
        if a.shape != want:
            raise CodecError(f"get() returned {name} with shape {a.shape}, expected {want}")
        if not np.all(np.isfinite(a)):
            raise CodecError(f"get() returned non-finite values in {name}")


def _kl(lp, lq):
    return float(np.sum(np.exp(lp) * (lp - lq)))


def evaluate_case(model, case, sandbox, log=print):
    """Run one case. Returns a dict with 'passed' and diagnostics."""
    prompt = list(case["prompt"])
    cont = list(case["continuation"])
    chunks = list(case.get("chunks", [len(prompt)]))
    assert sum(chunks) == len(prompt)
    P = len(prompt)
    res = {"id": case["id"], "prompt_len": P, "passed": False, "failures": []}

    kv, _ = model.prefill(prompt)
    exact = ExactCache(kv)
    ref_lp = [log_softmax(model.decode_step(tok, P + t, exact)) for t, tok in enumerate(cont)]

    sizes, kls, step_times = [], [], []

    def record_size(blob, T):
        sizes.append(len(blob) / budget(T))
        if len(blob) > budget(T) and len(res["failures"]) < 5:
            res["failures"].append(f"blob of {len(blob)} bytes with T={T} exceeds budget {budget(T)}")
        if len(blob) > MAX_BLOB_BYTES:
            raise CodecError("blob is unreasonably large")

    try:
        cp = sandbox.spawn(PREFILL_SECONDS)
        try:
            cp.init(N_LAYERS, N_HEADS, HEAD_DIM)
            start = 0
            for c in chunks:
                for layer in range(N_LAYERS):
                    cp.append(layer, kv[layer][0][:, start:start + c], kv[layer][1][:, start:start + c])
                start += c
            blob = cp.to_bytes()
            res["prefill_seconds"] = round(cp.used_seconds(), 3)
        finally:
            cp.close()
        record_size(blob, P)

        for t, tok in enumerate(cont):
            T = P + t + 1
            cp = sandbox.spawn(DECODE_STEP_SECONDS)
            try:
                cp.from_bytes(blob, N_LAYERS, N_HEADS, HEAD_DIM)

                def provider(layer, k, v):
                    cp.append(layer, k, v)
                    K, V = cp.get(layer)
                    _check_kv(K, V, T)
                    return K, V

                lq = log_softmax(model.decode_step(tok, P + t, provider))
                blob = cp.to_bytes()
                step_times.append(cp.used_seconds())
            finally:
                cp.close()
            record_size(blob, T)
            kls.append(_kl(ref_lp[t], lq))
    except (CodecError, ValueError, TypeError, struct.error) as e:
        where = "prefill" if not sizes else f"decode step {len(kls)}"
        res["failures"].append(f"codec error during {where}: {e}")
        return res

    res["mean_kl"] = float(np.mean(kls))
    res["max_kl"] = float(np.max(kls))
    res["max_budget_ratio"] = round(max(sizes), 4)
    res["max_step_seconds"] = round(max(step_times), 3)
    if res["mean_kl"] > MEAN_KL_MAX:
        res["failures"].append(f"mean KL {res['mean_kl']:.5f} > {MEAN_KL_MAX}")
    if res["max_kl"] > MAX_KL_MAX:
        res["failures"].append(f"max KL {res['max_kl']:.5f} > {MAX_KL_MAX}")
    res["passed"] = not res["failures"]
    return res


def check_interface(sandbox, seed=0):
    """Contract checks on synthetic data (no accuracy requirement)."""
    rng = np.random.default_rng(seed)
    L, H, D = N_LAYERS, N_HEADS, HEAD_DIM
    chunks = [1, 37, 200, 3]

    def kv(t):
        return (rng.standard_normal((H, t, D)).astype(np.float32) * 2,
                rng.standard_normal((H, t, D)).astype(np.float32))

    data = [[kv(c) for _ in range(L)] for c in chunks]
    cp = sandbox.spawn(PREFILL_SECONDS)
    try:
        cp.init(L, H, D)
        T = 0
        for ci, c in enumerate(chunks):
            T += c
            for layer in range(L):
                cp.append(layer, *data[ci][layer])
                _check_kv(*cp.get(layer), T)
        blob = cp.to_bytes()
    finally:
        cp.close()
    for _ in range(2):
        cp = sandbox.spawn(DECODE_STEP_SECONDS)
        try:
            cp.from_bytes(blob, L, H, D)
            T += 1
            for layer in range(L):
                cp.append(layer, *kv(1))
                _check_kv(*cp.get(layer), T)
            blob = cp.to_bytes()
        finally:
            cp.close()


def load_cases(path):
    with open(path) as f:
        return json.load(f)
