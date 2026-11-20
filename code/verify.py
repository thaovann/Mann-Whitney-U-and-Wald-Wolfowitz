"""
Verification of the from-scratch implementation.

scipy / statsmodels are imported HERE ONLY, to check our numbers.
They are never used inside mann_whitney.py or wald_wolfowitz.py.
"""
import itertools
import math
import random

from mann_whitney import mann_whitney_u, exact_u_distribution
from wald_wolfowitz import wald_wolfowitz, runs_distribution, count_runs, runs_mean_var
from examples import MW_EXAMPLES, WW_EXAMPLES

ok = True


def check(name, a, b, tol=1e-9):
    global ok
    good = abs(a - b) <= tol
    ok &= good
    print(f"  [{'OK ' if good else 'BAD'}] {name}: ours = {a:.10f}  reference = {b:.10f}")


print("1) Mann-Whitney vs scipy.stats.mannwhitneyu")
from scipy.stats import mannwhitneyu
for k, ex in MW_EXAMPLES.items():
    for alt in ("two-sided", "greater", "less"):
        ours = mann_whitney_u(ex["x"], ex["y"], alternative=alt)
        ref = mannwhitneyu(ex["x"], ex["y"], alternative=alt,
                           method="exact" if ours.method == "exact" else "asymptotic")
        check(f"{k} {alt:9} U1", ours.U1, ref.statistic)
        check(f"{k} {alt:9} p ", ours.p_value, ref.pvalue)

rng = random.Random(1)
for trial in range(200):
    n1, n2 = rng.randint(1, 15), rng.randint(1, 15)
    x = [rng.random() for _ in range(n1)]
    y = [rng.random() + 0.3 for _ in range(n2)]
    for m, sm in (("exact", "exact"), ("normal", "asymptotic")):
        a = mann_whitney_u(x, y, method=m).p_value
        b = mannwhitneyu(x, y, method=sm).pvalue
        if abs(a - b) > 1e-9:
            ok = False
            print("  BAD random trial", n1, n2, m, a, b)
print("  200 random exact+normal trials compared")

print("\n2) Exact distribution of U sums to 1 and matches brute force (n1=4, n2=5)")
prob = exact_u_distribution(4, 5)
check("sum", sum(prob), 1.0)
brute = [0] * 21
for xs in itertools.combinations(range(9), 4):
    ranks = [r + 1 for r in xs]
    brute[sum(ranks) - 10] += 1
for u in range(21):
    check(f"P(U={u:2})", prob[u], brute[u] / math.comb(9, 4))

print("\n3) Runs distribution vs brute-force enumeration (n1=5, n2=6)")
dist = runs_distribution(5, 6)
total = math.comb(11, 5)
counts = {}
for pos in itertools.combinations(range(11), 5):
    labels = ["X" if i in pos else "Y" for i in range(11)]
    r = count_runs(labels)
    counts[r] = counts.get(r, 0) + 1
for r in sorted(counts):
    check(f"P(R={r:2})", dist[r], counts[r] / total)
mean, var = runs_mean_var(5, 6)
check("E[R]", mean, sum(r * p for r, p in dist.items()))
check("Var[R]", var, sum((r - mean) ** 2 * p for r, p in dist.items()))

print("\n4) Wald-Wolfowitz normal approx vs statsmodels runstest_2samp")
try:
    from statsmodels.sandbox.stats.runs import runstest_2samp
    import warnings
    for k, ex in WW_EXAMPLES.items():
        ours = wald_wolfowitz(ex["x"], ex["y"], method="normal", alternative="two-sided",
                              continuity=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z, p = runstest_2samp(ex["x"], ex["y"], correction=False)
        check(f"{k} two-sided p (no continuity corr.)", ours.p_value, p, tol=1e-6)
except ImportError:
    print("  statsmodels not installed - skipped")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
