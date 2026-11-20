"""
Demo: Mann-Whitney U test + Wald-Wolfowitz runs test (from-scratch implementation).

Usage
-----
python demo.py                 # run all examples step by step
python demo.py --plots         # also save figures into ./figures
python demo.py --x 1,2,3 --y 4,5,6,7          # your own data (both tests)
python demo.py --x ... --y ... --alt greater  # one-sided Mann-Whitney
"""
import argparse
import os
import statistics as st

from examples import MW_EXAMPLES, WW_EXAMPLES, RANDOMNESS_EXAMPLE
from mann_whitney import mann_whitney_u, exact_u_distribution, u_by_pair_counting
from wald_wolfowitz import wald_wolfowitz, runs_distribution, runs_test_randomness

ALPHA = 0.05
LINE = "=" * 78


def decision(p):
    return f"p = {p:.4f} {'<' if p < ALPHA else '>='} {ALPHA} -> " + \
        ("REJECT H0" if p < ALPHA else "do NOT reject H0")


def describe(name, data):
    print(f"  {name:<14} n={len(data):<3} median={st.median(data):<9.3f} "
          f"mean={st.mean(data):<9.3f} sd={st.stdev(data):.3f}")


# --------------------------------------------------------------------------
def show_mann_whitney(key, ex, verbose_ranks=True):
    x, y = ex["x"], ex["y"]
    print(LINE)
    print(f"[{key}] {ex['title']}")
    print(LINE)
    print("Note:", ex["note"])
    describe(ex["x_name"] + " (X)", x)
    describe(ex["y_name"] + " (Y)", y)

    res = mann_whitney_u(x, y, alternative=ex["alternative"])

    if verbose_ranks and len(x) + len(y) <= 30:
        print("\n  Step 1-2: pooled & ranked data")
        pooled = sorted([(v, "X", r) for v, r in zip(x, res.ranks_x)] +
                        [(v, "Y", r) for v, r in zip(y, res.ranks_y)])
        print("   value  : " + " ".join(f"{v:>6g}" for v, _, _ in pooled))
        print("   group  : " + " ".join(f"{g:>6}" for _, g, _ in pooled))
        print("   rank   : " + " ".join(f"{r:>6g}" for _, _, r in pooled))

    print(f"\n  Step 3: R1 = {res.R1:g}, R2 = {res.R2:g}  (check: R1+R2 = "
          f"{res.R1 + res.R2:g} = N(N+1)/2 = {(res.n1 + res.n2) * (res.n1 + res.n2 + 1) / 2:g})")
    print(f"  Step 4: U1 = R1 - n1(n1+1)/2 = {res.U1:g};  U2 = n1*n2 - U1 = {res.U2:g}")
    print(f"          (pair counting check: #(x>y)+0.5#(x=y) = {u_by_pair_counting(x, y):g})")
    print(f"  Step 5: method = {res.method}, H1 = {res.alternative}")
    if res.method == "normal":
        print(f"          E[U] = {res.mean_U:g}, SD[U] = {res.sd_U:.4f} "
              f"(tie groups: {res.tie_groups}), Z = {res.z:.4f}")
    print(f"  Effect size: P(X > Y) = {res.cles:.3f}, r = {res.effect_r:.3f}")
    print("  Decision:", decision(res.p_value))
    return res


def show_wald_wolfowitz(key, ex):
    x, y = ex["x"], ex["y"]
    print(LINE)
    print(f"[{key}] {ex['title']}")
    print(LINE)
    print("Note:", ex["note"])
    describe(ex["x_name"] + " (X)", x)
    describe(ex["y_name"] + " (Y)", y)

    res = wald_wolfowitz(x, y)
    print(f"\n  Step 1-2: sorted labels: {res.sequence}")
    print(f"  Step 3  : runs = {' | '.join(res.runs)}")
    print(f"            R = {res.R}")
    print(f"  Step 4  : E[R] = {res.mean_R:.3f}, SD[R] = {res.sd_R:.3f}, method = {res.method}"
          + (f", Z = {res.z:.4f}" if res.z is not None else ""))
    if res.ties_between_groups:
        print(f"  Ties between groups: {res.ties_between_groups}, R in "
              f"[{res.R_min_ties}, {res.R_max_ties}] -> using conservative R = {res.R}")
    print("  Decision (H1: too few runs):", decision(res.p_value))

    mw = mann_whitney_u(x, y)
    print(f"  For comparison, Mann-Whitney (two-sided): U = {mw.U:g}, p = {mw.p_value:.4f}")
    return res


def show_randomness():
    ex = RANDOMNESS_EXAMPLE
    print(LINE)
    print("[RUNS-1] " + ex["title"])
    print(LINE)
    out = runs_test_randomness(ex["sequence"])
    print("  sequence:", out["labels"])
    print(f"  n1 = {out['n1']}, n2 = {out['n2']}, R = {out['R']}, "
          f"E[R] = {out['mean_R']:.3f}, SD[R] = {out['sd_R']:.3f}, method = {out['method']}")
    print("  Decision (two-sided):", decision(out["p_value"]))


# --------------------------------------------------------------------------
def save_plots(folder="figures"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(folder, exist_ok=True)
    c_x, c_y, c_obs = "#2a6fdb", "#e0782b", "#c0392b"

    # 1. exact null distribution of U for MW1
    ex = MW_EXAMPLES["MW1"]
    res = mann_whitney_u(ex["x"], ex["y"], alternative="greater")
    prob = exact_u_distribution(res.n1, res.n2)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(prob)), prob,
           color=[c_obs if u >= res.U1 else "#9aa5b1" for u in range(len(prob))])
    ax.set_title(f"Exact null distribution of U (n1={res.n1}, n2={res.n2}); "
                 f"red = P(U >= {res.U1:g}) = {res.p_value:.4f}")
    ax.set_xlabel("U"); ax.set_ylabel("P(U = u)")
    fig.tight_layout(); fig.savefig(os.path.join(folder, "mw1_null_distribution.png"), dpi=150)
    plt.close(fig)

    # 2. MW3 boxplot (outliers)
    ex = MW_EXAMPLES["MW3"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([ex["x"], ex["y"]], labels=[ex["x_name"], ex["y_name"]], vert=False)
    ax.set_xscale("log"); ax.set_xlabel("response time (ms, log scale)")
    ax.set_title("MW3: outliers distort the mean, not the ranks")
    fig.tight_layout(); fig.savefig(os.path.join(folder, "mw3_boxplot.png"), dpi=150)
    plt.close(fig)

    # 3. runs strip plots for WW1 and WW2
    for key in ("WW1", "WW2"):
        ex = WW_EXAMPLES[key]
        res = wald_wolfowitz(ex["x"], ex["y"])
        fig, ax = plt.subplots(figsize=(9, 2.4))
        ax.scatter(ex["x"], [0] * len(ex["x"]), color=c_x, s=70, label=ex["x_name"] + " (X)")
        ax.scatter(ex["y"], [0] * len(ex["y"]), color=c_y, s=70, marker="s",
                   label=ex["y_name"] + " (Y)")
        ax.set_yticks([]); ax.legend(loc="upper center", ncol=2, frameon=False,
                                     bbox_to_anchor=(0.5, 1.35))
        ax.set_title(f"{key}: sorted labels {res.sequence}  ->  R = {res.R} runs", pad=30)
        fig.tight_layout(); fig.savefig(os.path.join(folder, f"{key.lower()}_runs.png"), dpi=150)
        plt.close(fig)

    # 4. null distribution of R for WW1
    ex = WW_EXAMPLES["WW1"]
    res = wald_wolfowitz(ex["x"], ex["y"])
    dist = runs_distribution(res.n1, res.n2)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(dist), list(dist.values()),
           color=[c_obs if r <= res.R else "#9aa5b1" for r in dist])
    ax.axvline(res.mean_R, ls="--", color="k", lw=1)
    ax.set_title(f"Exact null distribution of R (n1={res.n1}, n2={res.n2}); "
                 f"red = P(R <= {res.R}) = {res.p_value:.5f}")
    ax.set_xlabel("number of runs R"); ax.set_ylabel("P(R = r)")
    fig.tight_layout(); fig.savefig(os.path.join(folder, "ww1_null_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"Figures saved in ./{folder}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", help="comma-separated sample X")
    ap.add_argument("--y", help="comma-separated sample Y")
    ap.add_argument("--alt", default="two-sided",
                    choices=["two-sided", "greater", "less"], help="H1 for Mann-Whitney")
    ap.add_argument("--plots", action="store_true", help="save figures")
    args = ap.parse_args()

    if args.x and args.y:
        x = [float(v) for v in args.x.split(",")]
        y = [float(v) for v in args.y.split(",")]
        print(mann_whitney_u(x, y, alternative=args.alt).summary())
        print()
        print(wald_wolfowitz(x, y).summary())
        return

    print("\n######## PART 1 - MANN-WHITNEY U TEST ########\n")
    for k, ex in MW_EXAMPLES.items():
        show_mann_whitney(k, ex)
        if k == "MW3":
            try:  # comparison only -- the t-test is NOT the topic of this project
                from scipy.stats import ttest_ind
                t = ttest_ind(ex["x"], ex["y"], equal_var=False)
                print(f"  For comparison, Welch t-test: t = {t.statistic:.3f}, p = {t.pvalue:.4f}")
            except ImportError:
                pass
        print()

    print("\n######## PART 2 - WALD-WOLFOWITZ RUNS TEST ########\n")
    for k, ex in WW_EXAMPLES.items():
        show_wald_wolfowitz(k, ex)
        print()
    show_randomness()

    if args.plots:
        save_plots()


if __name__ == "__main__":
    main()
