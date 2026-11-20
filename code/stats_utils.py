"""
Small helper functions written from scratch (no scipy).

- rank_with_ties : average ranks for tied values
- normal_cdf     : standard normal CDF using math.erf
- comb           : binomial coefficient (math.comb)
"""
import math


def normal_cdf(z):
    """Phi(z) = P(Z <= z) for Z ~ N(0, 1)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def comb(n, k):
    """Binomial coefficient C(n, k); 0 if k is out of range."""
    if k < 0 or k > n or n < 0:
        return 0
    return math.comb(n, k)


def rank_with_ties(values):
    """
    Assign ranks 1..N to `values`. Tied values receive the AVERAGE of the
    ranks they would occupy.

    Returns
    -------
    ranks      : list of float, rank of values[i]
    tie_groups : list of int, size t of every group of tied values (t >= 2)
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    tie_groups = []

    i = 0
    while i < n:
        j = i
        # extend j while the next value equals the current one
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # positions i..j (0-based) -> ranks i+1..j+1 ; average = (i+j)/2 + 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        t = j - i + 1
        if t > 1:
            tie_groups.append(t)
        i = j + 1
    return ranks, tie_groups
