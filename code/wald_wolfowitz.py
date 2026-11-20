"""
Wald-Wolfowitz runs test -- implemented from scratch.

(a) Two-sample version  (main topic)
    H0 : X and Y come from the same continuous distribution
    H1 : the distributions differ in ANY way (location, spread, shape)
    Idea: pool and sort the N = n1 + n2 observations, write the group label
    of each one (e.g. X X Y X Y Y ...) and count the number of RUNS R
    (maximal blocks of identical labels).
    If H0 is true the labels are well mixed -> many runs.
    If the distributions differ, labels cluster  -> FEW runs.
    => one-sided (left-tail) test: reject H0 when R is too small.

(b) One-sample runs test for randomness of a binary sequence
    (same distribution of R, two-sided by default).

Exact null distribution of R (n1 symbols of type A, n2 of type B):
    R = 2k     : P = 2 * C(n1-1, k-1) * C(n2-1, k-1) / C(N, n1)
    R = 2k + 1 : P = [C(n1-1, k) C(n2-1, k-1) + C(n1-1, k-1) C(n2-1, k)] / C(N, n1)
Normal approximation:
    mu_R    = 2 n1 n2 / N + 1
    sigma^2 = 2 n1 n2 (2 n1 n2 - N) / (N^2 (N - 1))
    Z = (R - mu_R + 0.5) / sigma_R    (left tail, continuity correction)
"""
import math
from dataclasses import dataclass, field

from stats_utils import comb, normal_cdf


# ---------------------------------------------------------------------------
# Runs counting and exact distribution
# ---------------------------------------------------------------------------
def count_runs(labels):
    """Number of runs in a sequence, e.g. 'XXYXYY' -> 4."""
    labels = list(labels)
    if not labels:
        return 0
    return 1 + sum(1 for a, b in zip(labels, labels[1:]) if a != b)


def split_runs(labels):
    """Return the runs themselves, e.g. 'XXYX' -> ['XX', 'Y', 'X']."""
    runs, current = [], [labels[0]]
    for lab in labels[1:]:
        if lab == current[-1]:
            current.append(lab)
        else:
            runs.append("".join(current))
            current = [lab]
    runs.append("".join(current))
    return runs


def runs_pmf(r, n1, n2):
    """Exact P(R = r) under H0 (all C(N, n1) arrangements equally likely)."""
    N = n1 + n2
    total = comb(N, n1)
    if r < 2 or r > 2 * min(n1, n2) + (1 if n1 != n2 else 0):
        return 0.0
    if r % 2 == 0:
        k = r // 2
        ways = 2 * comb(n1 - 1, k - 1) * comb(n2 - 1, k - 1)
    else:
        k = (r - 1) // 2
        ways = comb(n1 - 1, k) * comb(n2 - 1, k - 1) + comb(n1 - 1, k - 1) * comb(n2 - 1, k)
    return ways / total


def runs_distribution(n1, n2):
    """Dictionary {r: P(R = r)} for all possible r."""
    r_max = 2 * min(n1, n2) + (1 if n1 != n2 else 0)
    return {r: runs_pmf(r, n1, n2) for r in range(2, r_max + 1)}


def runs_mean_var(n1, n2):
    N = n1 + n2
    mean = 2.0 * n1 * n2 / N + 1.0
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - N) / (N ** 2 * (N - 1))
    return mean, var


def _p_value(R, n1, n2, alternative, method, continuity=True):
    """alternative: 'less' (too few runs), 'greater' (too many), 'two-sided'."""
    mean, var = runs_mean_var(n1, n2)
    sd = math.sqrt(var)
    z = None
    if method == "exact":
        dist = runs_distribution(n1, n2)
        p_le = sum(p for r, p in dist.items() if r <= R)
        p_ge = sum(p for r, p in dist.items() if r >= R)
    else:
        cc = 0.5 if continuity else 0.0
        z_le = (R - mean + cc) / sd          # for P(R <= r)
        z_ge = (R - mean - cc) / sd          # for P(R >= r)
        p_le = normal_cdf(z_le)
        p_ge = 1.0 - normal_cdf(z_ge)
        z = (R - mean) / sd
        if alternative == "less":
            z = z_le
        elif alternative == "greater":
            z = z_ge
    if alternative == "less":
        p = p_le
    elif alternative == "greater":
        p = p_ge
    else:
        p = min(1.0, 2.0 * min(p_le, p_ge))
    return p, z, mean, sd


# ---------------------------------------------------------------------------
# (a) Two-sample Wald-Wolfowitz test
# ---------------------------------------------------------------------------
@dataclass
class WaldWolfowitzResult:
    n1: int
    n2: int
    R: int
    mean_R: float
    sd_R: float
    method: str
    alternative: str
    p_value: float
    z: float = None
    sequence: str = ""
    runs: list = field(default_factory=list)
    ties_between_groups: int = 0
    R_min_ties: int = None   # fewest runs possible when mixed ties are re-ordered
    R_max_ties: int = None   # most runs possible

    def summary(self):
        lines = [
            "Wald-Wolfowitz runs test (two samples)",
            f"  n1 = {self.n1} (X), n2 = {self.n2} (Y), N = {self.n1 + self.n2}",
            f"  sorted labels : {self.sequence}",
            f"  runs          : {' | '.join(self.runs)}",
            f"  R = {self.R}   E[R] = {self.mean_R:.4f}   SD[R] = {self.sd_R:.4f}",
            f"  method = {self.method}, alternative = {self.alternative}",
        ]
        if self.z is not None:
            lines.append(f"  Z = {self.z:.4f}")
        if self.ties_between_groups:
            lines.append(f"  WARNING: {self.ties_between_groups} tie block(s) shared by X and Y;"
                         f" R can range from {self.R_min_ties} to {self.R_max_ties}."
                         f" Reported R = {self.R} (conservative = max).")
        lines.append(f"  p-value = {self.p_value:.6f}")
        return "\n".join(lines)


def _label_sequence(x, y, tie_strategy):
    """
    Sort the pooled sample and return the label string.
    Inside a block of equal values that contains both X and Y, the order is
    arbitrary. We can arrange it to give the MIN or the MAX number of runs.
    'conservative' -> MAX runs (largest p-value, safest decision).
    Returns (sequence, n_mixed_tie_blocks, seq_min, seq_max).
    """
    pooled = sorted([(v, "X") for v in x] + [(v, "Y") for v in y], key=lambda t: t[0])

    # group equal values
    blocks, i = [], 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        blocks.append([lab for _, lab in pooled[i:j + 1]])
        i = j + 1

    mixed = sum(1 for b in blocks if "X" in b and "Y" in b)

    def build(maximize):
        seq = []
        for b in blocks:
            nx, ny = b.count("X"), b.count("Y")
            prev = seq[-1] if seq else None
            if nx == 0 or ny == 0:
                seq.extend(b)
                continue
            if maximize:
                # alternate, starting with the label different from prev
                first = "Y" if prev == "X" else "X"
                if prev is None:
                    first = "X" if nx >= ny else "Y"
                cnt = {"X": nx, "Y": ny}
                cur = first
                while cnt["X"] > 0 and cnt["Y"] > 0:   # alternate X Y X Y ...
                    seq.append(cur)
                    cnt[cur] -= 1
                    cur = "Y" if cur == "X" else "X"
                for lab in ("X", "Y"):                  # leftovers (one label)
                    seq.extend([lab] * cnt[lab])
            else:
                # put the label equal to prev first -> fewest new runs
                first = prev if prev in ("X", "Y") else "X"
                second = "Y" if first == "X" else "X"
                seq.extend([first] * b.count(first) + [second] * b.count(second))
        return "".join(seq)

    s_min, s_max = build(False), build(True)
    if tie_strategy == "min":
        chosen = s_min
    else:  # 'conservative' / 'max'
        chosen = s_max
    return chosen, mixed, s_min, s_max


def wald_wolfowitz(x, y, method="auto", alternative="less",
                   tie_strategy="conservative", continuity=True):
    """
    Two-sample Wald-Wolfowitz runs test.

    Parameters
    ----------
    x, y        : the two independent samples
    method      : 'auto' | 'exact' | 'normal'
                  auto -> exact when n1 <= 20 and n2 <= 20, else normal
    alternative : 'less' (default, the classical WW test: too FEW runs)
                  'two-sided' or 'greater' are also available
    tie_strategy: 'conservative' (max runs) or 'min' when X and Y share values
    """
    x, y = list(x), list(y)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        raise ValueError("both samples must be non-empty")
    seq, mixed, s_min, s_max = _label_sequence(x, y, tie_strategy)
    R = count_runs(seq)

    if method == "auto":
        method = "exact" if (n1 <= 20 and n2 <= 20) else "normal"

    p, z, mean, sd = _p_value(R, n1, n2, alternative, method, continuity)
    return WaldWolfowitzResult(
        n1=n1, n2=n2, R=R, mean_R=mean, sd_R=sd, method=method,
        alternative=alternative, p_value=p, z=z, sequence=seq,
        runs=split_runs(seq), ties_between_groups=mixed,
        R_min_ties=count_runs(s_min), R_max_ties=count_runs(s_max),
    )


# ---------------------------------------------------------------------------
# (b) One-sample runs test for randomness
# ---------------------------------------------------------------------------
def runs_test_randomness(sequence, cutoff=None, method="auto",
                         alternative="two-sided"):
    """
    Runs test for randomness of one sequence.
    * If `sequence` is numeric, values are dichotomised around `cutoff`
      (default: the median); values equal to the cutoff are dropped.
    * If it is already binary (2 distinct symbols) it is used directly.
    """
    seq = list(sequence)
    symbols = sorted(set(seq), key=str)
    if len(symbols) > 2 or cutoff is not None:
        if cutoff is None:
            s = sorted(seq)
            m = len(s)
            cutoff = s[m // 2] if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2.0
        labels = ["+" if v > cutoff else "-" for v in seq if v != cutoff]
    else:
        labels = [str(v) for v in seq]
    a = labels[0]
    n1 = sum(1 for l in labels if l == a)
    n2 = len(labels) - n1
    if n2 == 0:
        raise ValueError("sequence contains only one symbol")
    R = count_runs(labels)
    if method == "auto":
        method = "exact" if (n1 <= 20 and n2 <= 20) else "normal"
    p, z, mean, sd = _p_value(R, n1, n2, alternative, method)
    return {"labels": "".join(labels), "cutoff": cutoff, "n1": n1, "n2": n2,
            "R": R, "mean_R": mean, "sd_R": sd, "z": z,
            "method": method, "alternative": alternative, "p_value": p}
