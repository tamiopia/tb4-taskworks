"""Development evaluation: same protocol, limits and thresholds as the hidden one.

    python /app/harness/run_eval.py                  # all development cases
    python /app/harness/run_eval.py --case dev-512   # one case
    python /app/harness/run_eval.py --no-isolate     # skip per-step sandboxing (faster)
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import evaluator as ev  # noqa: E402
from model import Model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", default="/app/submission/kvcache.py")
    ap.add_argument("--cases", default=os.path.join(os.path.dirname(HERE), "data", "dev_cases.json"))
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(HERE), "model", "dev_weights.npz"))
    ap.add_argument("--case", action="append", help="only run this case id (repeatable)")
    ap.add_argument("--no-isolate", action="store_true",
                    help="run the codec as the current user without the per-step cleanup")
    ap.add_argument("--json", help="write per-case results to this file")
    a = ap.parse_args()

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    isolate = not a.no_isolate
    if isolate and os.geteuid() != 0:
        print("not running as root: falling back to --no-isolate")
        isolate = False

    model = Model.load(a.weights)
    cases = ev.load_cases(a.cases)["cases"]
    if a.case:
        cases = [c for c in cases if c["id"] in a.case]
    print(f"budget: {ev.BUDGET_PER_TOKEN} * T + {ev.BUDGET_FIXED} bytes; "
          f"mean KL <= {ev.MEAN_KL_MAX}; max KL <= {ev.MAX_KL_MAX}")

    try:
        sb = ev.Sandbox(a.codec, isolate=isolate)
    except ev.CodecError as e:
        print(e)
        sys.exit(1)
    results = []
    try:
        try:
            ev.check_interface(sb)
            print("interface: ok")
        except (ev.CodecError, ValueError, TypeError) as e:
            print(f"interface: FAILED\n{e}")
        for case in cases:
            t0 = time.time()
            r = ev.evaluate_case(model, case, sb)
            r["wall_seconds"] = round(time.time() - t0, 1)
            results.append(r)
            status = "PASS" if r["passed"] else "FAIL"
            metrics = {k: r[k] for k in ("mean_kl", "max_kl", "max_budget_ratio",
                                         "prefill_seconds", "max_step_seconds") if k in r}
            print(f"{case['id']}: {status} {json.dumps(metrics)}")
            for f in r["failures"]:
                print("    " + f)
    finally:
        sb.close()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)
    n = sum(r["passed"] for r in results)
    print(f"{n}/{len(results)} cases passed")
    groups = {}
    for case, r in zip(cases, results):
        groups.setdefault(case.get("group", "prefill"), []).append(r["passed"])
    for g, ok in groups.items():
        print(f"criterion {g}: {'PASS' if all(ok) else 'FAIL'} ({sum(ok)}/{len(ok)} cases)")
    sys.exit(0 if n == len(results) else 1)


if __name__ == "__main__":
    main()
