# Mann-Whitney U test + Wald-Wolfowitz runs test (from scratch)

Python implementation for Topic 1 of the Applied Statistics presentation.
No existing test functions are used in the implementation (no `scipy.stats.mannwhitneyu`,
no statsmodels runs test). Only the standard library (`math`, `dataclasses`) is used;
matplotlib is used only to draw figures.

## Files

| File | Contents |
|---|---|
| `stats_utils.py` | average ranks with ties, standard normal CDF (via `math.erf`), binomial coefficient |
| `mann_whitney.py` | `mann_whitney_u(x, y, alternative, method)`: U statistic, exact distribution (dynamic programming), tie-corrected normal approximation, effect sizes |
| `wald_wolfowitz.py` | `wald_wolfowitz(x, y, method, alternative)`: runs, exact distribution of R, normal approximation, tie handling; `runs_test_randomness(seq)`: one-sample runs test |
| `examples.py` | data for the 3 Mann-Whitney and 3 Wald-Wolfowitz examples (+1 randomness example) |
| `demo.py` | step-by-step demo |
| `verify.py` | checks against brute-force enumeration and scipy/statsmodels (verification only) |

## Run

```
pip install matplotlib          # only needed for --plots
python demo.py                  # all examples, step by step
python demo.py --plots          # also save figures to ./figures
python demo.py --x 3,5,8,9 --y 1,2,4 --alt greater   # your own data
python verify.py                # needs scipy + statsmodels
```

## Quick use

```python
from mann_whitney import mann_whitney_u
from wald_wolfowitz import wald_wolfowitz

print(mann_whitney_u([78, 85, 92], [70, 75, 82], alternative="greater").summary())
print(wald_wolfowitz([49.8, 50.1, 49.9], [47.5, 52.6, 48.2]).summary())
```
