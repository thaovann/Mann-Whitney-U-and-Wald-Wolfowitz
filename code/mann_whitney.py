"""
Mann-Whitney U test (Wilcoxon rank-sum test) -- implemented from scratch.

H0 : the two populations have the same distribution
     (P(X > Y) = P(Y > X)  /  no location shift)
H1 : two-sided   -> distributions differ in location
     'greater'   -> X tends to be larger than Y
     'less'      -> X tends to be smaller than Y

Algorithm
---------
1. Pool the two samples (n1 + n2 = N) and rank them 1..N (average rank for ties).
2. R1 = sum of ranks of sample X.
3. U1 = R1 - n1(n1+1)/2      (= number of pairs (x, y) with x > y, ties count 1/2)
   U2 = n1*n2 - U1
4. p-value
   * exact  : null distribution of U obtained by counting all C(N, n1)
              equally-likely rank allocations with a recursion (no ties only)
   * normal : Z = (U1 - mu - cc) / sigma, mu = n1 n2 / 2,
              sigma^2 = n1 n2 / 12 * [(N + 1) - sum(t^3 - t) / (N (N - 1))]
"""
import math
from dataclasses import dataclass, field

from stats_utils import normal_cdf, rank_with_ties


@dataclass
class MannWhitneyResult:
    n1: int
    n2: int
    R1: float
    R2: float
    U1: float
    U2: float
    U: float                 # min(U1, U2), the value used with critical-value tables
    method: str              # 'exact' or 'normal'
    alternative: str
    p_value: float
    z: float = None          # only for the normal approximation
    mean_U: float = None
    sd_U: float = None
    tie_groups: list = field(default_factory=list)
    effect_r: float = None   # r = |Z| / sqrt(N)
    cles: float = None       # common-language effect size  P(X > Y) + 0.5 P(X = Y)
    ranks_x: list = None
    ranks_y: list = None

    def summary(self):
        lines = [
            "Mann-Whitney U test",
            f"  n1 = {self.n1}, n2 = {self.n2}",
            f"  R1 = {self.R1:g}, R2 = {self.R2:g}",
            f"  U1 = {self.U1:g}, U2 = {self.U2:g}, U = min = {self.U:g}",
            f"  method = {self.method}, alternative = {self.alternative}",
        ]
        if self.method == "normal":
            lines.append(f"  E[U] = {self.mean_U:g}, SD[U] = {self.sd_U:.4f}, Z = {self.z:.4f}")
        if self.tie_groups:
            lines.append(f"  tie groups (sizes) = {self.tie_groups}")
        lines.append(f"  p-value = {self.p_value:.6f}")
        lines.append(f"  effect size: CLES = P(X>Y) = {self.cles:.3f}"
                     + (f", r = {self.effect_r:.3f}" if self.effect_r is not None else ""))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exact null distribution of U
# ---------------------------------------------------------------------------
def exact_u_distribution(n1, n2):
    """
    Return a list `prob` with prob[u] = P(U1 = u | H0), u = 0..n1*n2.

    Recursion on the LARGEST observation of the pooled sample:
      * if it belongs to X it beats all n2 Y's  -> contributes n2 to U1
            c(u; n1, n2) += c(u - n2; n1 - 1, n2)
      * if it belongs to Y it adds nothing
            c(u; n1, n2) += c(u; n1, n2 - 1)
    c(u; n1, n2) counts arrangements, total = C(n1 + n2, n1).
    Implemented bottom-up (dynamic programming).
    """
    # table[i][j] = list of counts for sample sizes (i, j)
    table = [[None] * (n2 + 1) for _ in range(n1 + 1)]
    for i in range(n1 + 1):
        for j in range(n2 + 1):
            if i == 0 or j == 0:
                table[i][j] = [1]          # only u = 0 possible, one arrangement
                continue
            size = i * j + 1
            counts = [0] * size
            from_x = table[i - 1][j]       # largest obs is an X -> shift by j
            for u, c in enumerate(from_x):
                counts[u + j] += c
            from_y = table[i][j - 1]       # largest obs is a Y -> no shift
            for u, c in enumerate(from_y):
                counts[u] += c
            table[i][j] = counts
    counts = table[n1][n2]
    total = math.comb(n1 + n2, n1)
    return [c / total for c in counts]


def _exact_p(u1, n1, n2, alternative):
    prob = exact_u_distribution(n1, n2)
    u1 = int(round(u1))
    p_le = sum(prob[: u1 + 1])          # P(U1 <= u1)
    p_ge = sum(prob[u1:])               # P(U1 >= u1)
    if alternative == "greater":
        return p_ge
    if alternative == "less":
        return p_le
    return min(1.0, 2.0 * min(p_le, p_ge))


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def mann_whitney_u(x, y, alternative="two-sided", method="auto",
                   continuity=True):
    """
    Mann-Whitney U test for two independent samples.

    Parameters
    ----------
    x, y        : sequences of numbers (sample 1 and sample 2)
    alternative : 'two-sided' | 'greater' | 'less'   (about X relative to Y)
    method      : 'auto' | 'exact' | 'normal'
                  auto -> exact when there are no ties and n1, n2 <= 20,
                          otherwise the normal approximation
    continuity  : apply the 0.5 continuity correction in the normal method
    """
    x, y = list(x), list(y)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        raise ValueError("both samples must be non-empty")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")

    N = n1 + n2
    ranks, tie_groups = rank_with_ties(x + y)
    ranks_x, ranks_y = ranks[:n1], ranks[n1:]
    R1, R2 = sum(ranks_x), sum(ranks_y)

    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)

    if method == "auto":
        method = "exact" if (not tie_groups and n1 <= 20 and n2 <= 20) else "normal"
    if method == "exact" and tie_groups:
        raise ValueError("exact method assumes no ties; use method='normal'")

    mean_U = n1 * n2 / 2.0
    tie_term = sum(t ** 3 - t for t in tie_groups) / (N * (N - 1))
    var_U = n1 * n2 / 12.0 * ((N + 1) - tie_term)
    sd_U = math.sqrt(var_U)

    z = None
    if method == "exact":
        p = _exact_p(U1, n1, n2, alternative)
        # z still reported (no continuity) to compute effect size r
        z_for_r = (U1 - mean_U) / sd_U if sd_U > 0 else 0.0
    else:
        cc = 0.5 if continuity else 0.0
        if sd_U == 0:
            z, p = 0.0, 1.0
        elif alternative == "greater":
            z = (U1 - mean_U - cc) / sd_U
            p = 1.0 - normal_cdf(z)
        elif alternative == "less":
            z = (U1 - mean_U + cc) / sd_U
            p = normal_cdf(z)
        else:
            z = (abs(U1 - mean_U) - cc) / sd_U
            z = max(z, 0.0)
            p = min(1.0, 2.0 * (1.0 - normal_cdf(z)))
        z_for_r = z

    return MannWhitneyResult(
        n1=n1, n2=n2, R1=R1, R2=R2, U1=U1, U2=U2, U=U,
        method=method, alternative=alternative, p_value=p,
        z=z, mean_U=mean_U, sd_U=sd_U, tie_groups=tie_groups,
        effect_r=abs(z_for_r) / math.sqrt(N),
        cles=U1 / (n1 * n2),
        ranks_x=ranks_x, ranks_y=ranks_y,
    )


def u_by_pair_counting(x, y):
    """Alternative definition (useful for teaching / checking):
    U1 = #{(xi, yj): xi > yj} + 0.5 * #{(xi, yj): xi == yj}."""
    return sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in x for b in y)
