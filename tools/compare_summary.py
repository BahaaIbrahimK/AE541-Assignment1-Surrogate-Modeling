"""Compare a fresh run's summary.json with the committed one.

Usage: python tools/compare_summary.py reference.json new.json

Values at round-off level (|x| < 1e-9) are skipped, because they depend on the
BLAS library and CPU, not on the method. Everything else must agree to within
a relative tolerance of 1 %, and the selected polynomial degrees exactly.
"""
import json
import math
import sys

RTOL = 1e-2
ROUNDOFF = 1e-9
SKIP = {"platform", "abnormal_terminations", "true_function_nfev"}


def compare(ref, new, path, problems):
    if isinstance(ref, dict):
        for key in ref:
            if key in SKIP:
                continue
            if key not in new:
                problems.append(f"{path}/{key}: missing in new run")
            else:
                compare(ref[key], new[key], f"{path}/{key}", problems)
    elif isinstance(ref, list):
        if len(ref) != len(new):
            problems.append(f"{path}: length {len(ref)} -> {len(new)}")
        for i, (r, n) in enumerate(zip(ref, new)):
            compare(r, n, f"{path}[{i}]", problems)
    elif isinstance(ref, bool) or isinstance(ref, str) or ref is None:
        if ref != new:
            problems.append(f"{path}: {ref!r} -> {new!r}")
    elif isinstance(ref, int) and path.endswith("selected_degree"):
        if ref != new:
            problems.append(f"{path}: degree {ref} -> {new}")
    else:
        if abs(ref) < ROUNDOFF and abs(new) < ROUNDOFF:
            return
        if not math.isclose(ref, new, rel_tol=RTOL, abs_tol=ROUNDOFF):
            problems.append(f"{path}: {ref:.6g} -> {new:.6g}")


def main():
    with open(sys.argv[1]) as fh:
        ref = json.load(fh)
    with open(sys.argv[2]) as fh:
        new = json.load(fh)
    problems = []
    compare(ref, new, "", problems)
    if problems:
        print("Results differ from the committed ones:")
        print("\n".join(f"  {p}" for p in problems))
        sys.exit(1)
    print("Fresh run matches the committed results.")


if __name__ == "__main__":
    main()
