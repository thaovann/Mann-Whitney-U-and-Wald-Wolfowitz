"""
Datasets used in the presentation (illustrative data).
"""
import random

# ---------------------------------------------------------------- Mann-Whitney
MW_EXAMPLES = {
    "MW1": {
        "title": "Teaching method: new (X) vs traditional (Y) exam scores",
        "x": [78, 85, 92, 88, 95, 81, 90],
        "y": [70, 75, 82, 68, 79, 73],
        "x_name": "New method", "y_name": "Traditional",
        "alternative": "greater",
        "note": "Small samples, no ties -> exact distribution of U. "
                "H1: new method gives higher scores.",
    },
    "MW2": {
        "title": "Customer satisfaction (Likert 1-5): Store A (X) vs Store B (Y)",
        "x": [4, 5, 3, 4, 5, 4, 3, 5, 4, 4, 2, 5],
        "y": [3, 2, 3, 4, 2, 3, 1, 3, 4, 2, 3, 3],
        "x_name": "Store A", "y_name": "Store B",
        "alternative": "two-sided",
        "note": "Ordinal data with many ties -> average ranks + tie-corrected "
                "normal approximation. A t-test is not appropriate for ordinal scores.",
    },
    "MW3": {
        "title": "Server response time (ms): Server A (X) vs Server B (Y)",
        "x": [145, 152, 160, 149, 158, 171, 155, 163, 150, 168],
        "y": [118, 125, 131, 122, 2950, 128, 135, 120, 3100, 127],
        "x_name": "Server A", "y_name": "Server B",
        "alternative": "two-sided",
        "note": "Heavy outliers in B: mean(B) > mean(A) but most B values are smaller. "
                "The t-test is misled, Mann-Whitney is robust.",
    },
}

# ---------------------------------------------------------- Wald-Wolfowitz
def _ww3_data():
    rng = random.Random(2026)
    # same mean (= 10) but different SHAPE: exponential vs normal
    x = [round(rng.expovariate(1 / 10.0), 2) for _ in range(30)]
    y = [round(rng.gauss(10.0, 1.5), 2) for _ in range(35)]
    return x, y


_x3, _y3 = _ww3_data()

WW_EXAMPLES = {
    "WW1": {
        "title": "Filling machines: new (X) vs worn (Y) -- same centre, different spread",
        "x": [49.8, 50.1, 49.9, 50.0, 50.2, 49.95, 50.05, 50.15],
        "y": [47.5, 52.6, 48.2, 51.9, 47.9, 52.3, 48.8, 51.4],
        "x_name": "New machine", "y_name": "Worn machine",
        "note": "Medians are almost equal, so Mann-Whitney finds nothing; "
                "the runs test detects the difference in variability.",
    },
    "WW2": {
        "title": "Reaction time (s): drug group (X) vs placebo (Y)",
        "x": [12.1, 13.4, 11.8, 14.2, 12.9, 13.0, 11.5],
        "y": [15.3, 14.8, 16.1, 13.9, 15.7, 16.4, 14.5, 15.0],
        "x_name": "Drug", "y_name": "Placebo",
        "note": "Difference in location -> the labels form few long runs.",
    },
    "WW3": {
        "title": "Waiting time (min): exponential process (X, n=30) vs scheduled process (Y, n=35)",
        "x": _x3,
        "y": _y3,
        "x_name": "Process X", "y_name": "Process Y",
        "note": "Large samples -> normal approximation. Both means are about 10 "
                "but the shapes differ (skewed vs symmetric).",
    },
}

# One-sample runs test for randomness (bonus example)
RANDOMNESS_EXAMPLE = {
    "title": "Quality control: defective (D) / good (G) items in production order",
    "sequence": list("GGGGGGDDDGGGGGGGGDDDDGGGGGGGDDGGGG"),
    "note": "Defects arrive in clusters -> too few runs -> not random.",
}
