"""Rebuttal statistics: bootstrap CIs, McNemar tests, and perception->solving linkage.

Run from the repo root: python result_visualizations/rebuttal_analysis.py

Analysis 1: 95% bootstrap CIs for every accuracy in the rebuttal tables and
exact McNemar tests for the paired comparisons (same 500 puzzles per pair).

Analysis 2: per-puzzle join of object-detection (perception) and solving runs
on identical board types, reporting solving accuracy conditional on perception
quality (Fisher exact test on the pooled 2x2 table).
"""
import glob
import json
import math
import random

SOLVE_DIR = "evaluation/results/test/all"
OD_DIR = "evaluation/results/object_detection/test/all"

MODELS = {
    "Qwen3.5-397B-A17B-AWQ": "Qwen 397B",
    "Qwen3.5-27B": "Qwen 27B",
    "gemma-4-31B-it": "Gemma 4 31B",
}
# every model with joinable OD(default) + solving(prompt_engineering) runs
LINKAGE_MODELS = {
    **MODELS,
    "Qwen3-VL-235B-A22B-Thinking-FP8": "Qwen3-VL 235B",
    "GLM-4.6V": "GLM-4.6V",
    "gemma-3-27b-it": "Gemma 3 27B",
    "Llama-4-Scout-17B-16E-Instruct": "Llama-4 Scout",
    "Mistral-Small-3.2-24B-Instruct-2506": "Mistral-S 3.2 24B",
}
LADDER = ["original", "grid_lines_only", "coordinate_grid_no_labels", "coordinate_grid"]
OD_CORE = ["original", "start_end_marked", "coordinate_grid",
           "coordinate_grid_and_start_end_marked", "path_cell_annotated", "text"]


def load_individual(base, model, board_type, prompt, timestamp_filter=None):
    files = sorted(glob.glob(f"{base}/{model}/{board_type}-B_{prompt}-P_*_stats_individual.json"))
    if timestamp_filter == "new":
        files = [f for f in files if "-P_202607" in f]
    elif timestamp_filter == "old":
        files = [f for f in files if "-P_202607" not in f]
    if not files:
        return None
    return json.load(open(files[-1]))


def to_bool(v):
    if isinstance(v, str):
        return v.strip().lower() == "true" or v.strip() == "1"
    return bool(int(v))


def solve_outcomes(model, board_type):
    d = load_individual(SOLVE_DIR, model, board_type, "prompt_engineering")
    if d is None:
        return None
    return {pid: to_bool(it["is_valid"]) for pid, it in d.items()}


def od_outcomes(model, board_type, timestamp_filter):
    d = load_individual(OD_DIR, model, board_type, "default", timestamp_filter)
    if d is None:
        return None
    return {pid: (to_bool(it["is_valid"]), float(it["valid_fraction"])) for pid, it in d.items()}


def bootstrap_ci(outcomes, n_boot=10000, seed=42):
    rng = random.Random(seed)
    xs = list(outcomes)
    n = len(xs)
    means = sorted(sum(rng.choices(xs, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def mcnemar_exact(pairs):
    """pairs: list of (outcome_a, outcome_b) booleans for the same puzzle."""
    b = sum(1 for a, c in pairs if a and not c)  # A only
    c = sum(1 for a, cc in pairs if cc and not a)  # B only
    n = b + c
    if n == 0:
        return b, c, 1.0
    # exact two-sided binomial test, p0 = 0.5
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return b, c, min(1.0, 2 * p)


def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact for [[a, b], [c, d]] via hypergeometric enumeration."""
    row1, row2, col1 = a + b, c + d, a + c
    n = row1 + row2

    def hyper(x):
        return (math.comb(col1, x) * math.comb(n - col1, row1 - x)) / math.comb(n, row1)

    lo, hi = max(0, row1 - (n - col1)), min(row1, col1)
    p_obs = hyper(a)
    return sum(p for p in (hyper(x) for x in range(lo, hi + 1)) if p <= p_obs * (1 + 1e-9))


def fmt_ci(acc, lo, hi):
    return f"{acc:.3f} [{lo:.3f}, {hi:.3f}]"


def self_test():
    b, c, p = mcnemar_exact([(True, False)] * 5 + [(False, True)] * 5)
    assert (b, c) == (5, 5) and abs(p - 1.0) < 1e-9
    _, _, p = mcnemar_exact([(True, False)] * 15 + [(False, True)] * 2)
    assert p < 0.005  # 15 vs 2 discordant is clearly significant
    p = fisher_exact(10, 10, 10, 10)
    assert abs(p - 1.0) < 1e-9
    lo, hi = bootstrap_ci([True] * 50 + [False] * 50, n_boot=2000)
    assert lo < 0.5 < hi


def main():
    self_test()

    print("=" * 100)
    print("A. SOLVING LADDER with 95% bootstrap CIs (n=500, prompt_engineering)")
    print("=" * 100)
    solve = {}
    for model, mname in MODELS.items():
        solve[model] = {}
        cells = []
        for bt in LADDER:
            out = solve_outcomes(model, bt)
            solve[model][bt] = out
            if out is None:
                cells.append(f"{bt}: —")
                continue
            xs = list(out.values())
            acc = sum(xs) / len(xs)
            lo, hi = bootstrap_ci(xs)
            cells.append(f"{bt}: {fmt_ci(acc, lo, hi)}")
        print(f"\n{mname}")
        for cell in cells:
            print(f"  {cell}")

    print()
    print("=" * 100)
    print("B. PAIRED McNEMAR TESTS on the ladder (exact, two-sided; b/c = discordant counts)")
    print("=" * 100)
    comparisons = [
        ("coordinate_grid", "grid_lines_only", "(c) vs grid_lines_only  [label effect]"),
        ("grid_lines_only", "original", "grid_lines_only vs original  [grid-line effect]"),
        ("grid_lines_only", "coordinate_grid_no_labels", "grid_lines_only vs no_labels  [gutter effect]"),
        ("coordinate_grid", "original", "(c) vs original  [total effect]"),
    ]
    for model, mname in MODELS.items():
        print(f"\n{mname}")
        for bt_a, bt_b, label in comparisons:
            oa, ob = solve[model].get(bt_a), solve[model].get(bt_b)
            if not oa or not ob:
                continue
            ids = sorted(set(oa) & set(ob))
            pairs = [(oa[i], ob[i]) for i in ids]
            b, c, p = mcnemar_exact(pairs)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {label:55s} b={b:3d} c={c:3d}  p={p:.4f} {sig}")

    print()
    print("=" * 100)
    print("C. OD BUGGY vs FIXED PROMPT (Qwen 397B) with CIs and paired McNemar")
    print("=" * 100)
    model = "Qwen3.5-397B-A17B-AWQ"
    for bt in OD_CORE:
        old = od_outcomes(model, bt, "old")
        new = od_outcomes(model, bt, "new")
        if not old or not new:
            print(f"{bt}: missing runs")
            continue
        xs_old = [v[0] for v in old.values()]
        xs_new = [v[0] for v in new.values()]
        lo_o, hi_o = bootstrap_ci(xs_old)
        lo_n, hi_n = bootstrap_ci(xs_new)
        ids = sorted(set(old) & set(new))
        b, c, p = mcnemar_exact([(new[i][0], old[i][0]) for i in ids])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"{bt:40s} buggy {fmt_ci(sum(xs_old)/len(xs_old), lo_o, hi_o)}   "
              f"fixed {fmt_ci(sum(xs_new)/len(xs_new), lo_n, hi_n)}   "
              f"fixed-only={b} buggy-only={c} p={p:.4f} {sig}")

    print()
    print("=" * 100)
    print("D. PERCEPTION -> SOLVING LINKAGE (per-puzzle join, same board type)")
    print("   OD run: fixed prompt where available (397B core types), else original buggy-prompt run")
    print("=" * 100)
    summary = []
    for model, mname in LINKAGE_MODELS.items():
        pooled = []  # (od_fully_valid, od_fraction, solved)
        per_bt = {}
        for bt in OD_CORE:
            so = solve.get(model, {}).get(bt) or solve_outcomes(model, bt)
            od = od_outcomes(model, bt, "new") or od_outcomes(model, bt, "old")
            if not so or not od:
                continue
            ids = sorted(set(so) & set(od))
            rows = [(od[i][0], od[i][1], so[i]) for i in ids]
            per_bt[bt] = rows
            pooled.extend(rows)
        if not pooled:
            print(f"\n{mname}: no joinable runs")
            continue
        a = sum(1 for f, _, s in pooled if f and s)        # OD correct, solved
        b = sum(1 for f, _, s in pooled if f and not s)    # OD correct, unsolved
        c = sum(1 for f, _, s in pooled if not f and s)    # OD wrong, solved
        d = sum(1 for f, _, s in pooled if not f and not s)
        acc_odok = a / (a + b) if a + b else float("nan")
        acc_odbad = c / (c + d) if c + d else float("nan")
        p = fisher_exact(a, b, c, d)
        print(f"\n{mname}  (n={len(pooled)} puzzle x board-type pairs over {len(per_bt)} board types)")
        print(f"  P(solved | board fully perceived)     = {acc_odok:.3f}  (n={a+b})")
        print(f"  P(solved | board not fully perceived) = {acc_odbad:.3f}  (n={c+d})")
        rr = acc_odok / acc_odbad if acc_odbad > 0 else float("inf")
        print(f"  relative risk = {rr:.2f}x   Fisher exact p = {p:.2e}")
        print("  stratified by board type — P(solved | OD ok) vs P(solved | OD wrong)  (n_ok/n_wrong):")
        for bt, rows in per_bt.items():
            ok = [s for f, _, s in rows if f]
            bad = [s for f, _, s in rows if not f]
            if ok and bad:
                print(f"    {bt:40s} {sum(ok)/len(ok):.2f} vs {sum(bad)/len(bad):.2f}  ({len(ok)}/{len(bad)})")
        print("  solving accuracy by OD cell-level fraction:")
        bins = [(0.0, 0.7), (0.7, 0.9), (0.9, 0.999), (0.999, 1.01)]
        for lo_b, hi_b in bins:
            sel = [s for _, fr, s in pooled if lo_b <= fr < hi_b]
            if sel:
                print(f"    OD fraction [{lo_b:.1f}, {hi_b if hi_b <= 1 else 1.0:{'.1f' if hi_b <= 1 else '.1f'}}): "
                      f"acc={sum(sel)/len(sel):.3f} (n={len(sel)})")
        txt = per_bt.get("text", [])
        t_ok = [s for f, _, s in txt if f]
        t_bad = [s for f, _, s in txt if not f]
        summary.append((mname, len(per_bt), acc_odok, acc_odbad, rr, p,
                        sum(t_ok) / len(t_ok) if t_ok else float("nan"), len(t_ok),
                        sum(t_bad) / len(t_bad) if t_bad else float("nan"), len(t_bad)))

    print()
    print("=" * 100)
    print("E. CROSS-MODEL SUMMARY: pooled linkage + text-representation-only stratum")
    print("=" * 100)
    print(f"{'model':18s} {'#bts':>4s} {'P(s|OD ok)':>10s} {'P(s|OD bad)':>11s} {'RR':>6s} {'Fisher p':>9s}"
          f" | {'text: P(s|ok)':>13s} {'(n)':>6s} {'P(s|bad)':>9s} {'(n)':>6s}")
    for mname, nbt, aok, abad, rr, p, tok, ntok, tbad, ntbad in summary:
        print(f"{mname:18s} {nbt:4d} {aok:10.3f} {abad:11.3f} {rr:6.2f} {p:9.1e}"
              f" | {tok:13.3f} {ntok:6d} {tbad:9.3f} {ntbad:6d}")


if __name__ == "__main__":
    main()
