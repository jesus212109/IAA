"""
=============================================================================
HYPERGEOMETRIC PROBABILITY ANALYSIS
Practice 3: Evaluation Methodology and Scientific Rigor
Subject: Introduction to Machine Learning
3rd Year Computer Engineering - 2025/2026
=============================================================================

This script uses the hypergeometric distribution (scipy.stats.hypergeom)
to compute the exact probability of obtaining 0 positive cases in a random
test split, assuming:
    - Population:          N = 1000 patients
    - Positive class (K):  20 patients (2% of population)
    - Test set size (n):   200 patients (test_size = 20%)

The hypergeometric distribution models the number of successes (positives)
observed when drawing n samples WITHOUT replacement from a finite population
of N elements containing K successes.

PMF formula:
    P(X = k) = C(K, k) * C(N-K, n-k) / C(N, n)

where C(a, b) denotes the binomial coefficient "a choose b".

Result interpretation:
    P(X = 0) > 0  →  there is a non-negligible probability that a random
    split produces a test set with ZERO positive examples, which would make
    metrics such as F1-score, Recall, and AUC-ROC undefined or trivially zero.
=============================================================================
"""

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------
N = 1000   # Total population (number of patients)
K = 20     # Total number of positives in the population (2% of N)
n = 200    # Test set size (20% of N, i.e. test_size=0.20)

# ---------------------------------------------------------------------------
# HYPERGEOMETRIC DISTRIBUTION SETUP
# ---------------------------------------------------------------------------
# scipy parametrization: hypergeom(M, n, N) where
#   M = population size       → our N
#   n = number of successes   → our K
#   N = number of draws       → our n
# We follow scipy's convention to avoid confusion:
rv = stats.hypergeom(M=N, n=K, N=n)

# ---------------------------------------------------------------------------
# PROBABILITY CALCULATIONS
# ---------------------------------------------------------------------------
p_zero = rv.pmf(0)      # P(X = 0): zero positives in test set
p_one  = rv.pmf(1)      # P(X = 1): exactly one positive in test set
p_two  = rv.pmf(2)      # P(X = 2): exactly two positives in test set

# Expected number of positives in test
expected_positives = rv.mean()   # = n * K / N = 200 * 20 / 1000 = 4.0

# Standard deviation of the hypergeometric distribution
std_positives = rv.std()

# Cumulative probability of <= 1 positive (worst-case evaluation scenarios)
p_at_most_one = rv.cdf(1)

# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
separator = "=" * 70

print(separator)
print("  HYPERGEOMETRIC PROBABILITY ANALYSIS")
print("  Random Split with Imbalanced Classes (2% positive rate)")
print(separator)

print("\n[PARAMETERS]")
print(f"  Population size (N)            : {N} patients")
print(f"  Total positives (K)            : {K} ({100.0 * K / N:.1f}% of population)")
print(f"  Test set size (n)              : {n} patients (test_size = {100.0 * n / N:.0f}%)")
print(f"  Expected positives in test     : {expected_positives:.4f}")
print(f"  Std. dev. of positives in test : {std_positives:.4f}")

print("\n[PROBABILITY MASS FUNCTION — P(X = k)]")
print(f"  P(X = 0) — zero positives in test  : {p_zero:.6f}  ({100.0 * p_zero:.4f}%)")
print(f"  P(X = 1) — one positive in test     : {p_one:.6f}  ({100.0 * p_one:.4f}%)")
print(f"  P(X = 2) — two positives in test    : {p_two:.6f}  ({100.0 * p_two:.4f}%)")
print(f"  P(X <= 1) — at most one positive    : {p_at_most_one:.6f}  ({100.0 * p_at_most_one:.4f}%)")

print("\n[ANALYSIS]")
print(
    f"  With a random split (test_size=20%), there is a {100.0 * p_zero:.4f}% probability\n"
    f"  that the test set contains ZERO positive examples.\n"
    f"\n"
    f"  Furthermore, there is a {100.0 * p_at_most_one:.4f}% probability that the test set\n"
    f"  has AT MOST one positive example.\n"
    f"\n"
    f"  CONSEQUENCE: When X = 0, the following metrics are mathematically\n"
    f"  undefined or trivially zero for the positive class:\n"
    f"      - Recall         (division by zero: 0 true positives / 0 actual positives)\n"
    f"      - F1-score       (requires Recall to be defined)\n"
    f"      - AUC-ROC        (constant predictions, area is 0.5 or undefined)\n"
    f"\n"
    f"  This formally demonstrates that evaluating a model trained on\n"
    f"  imbalanced data without STRATIFICATION invalidates the analysis.\n"
    f"  The stratification constraint guarantees that every test split\n"
    f"  contains exactly round(K * test_size) = {round(K * n / N)} positive examples."
)

print("\n[LATEX OUTPUT — for inclusion in the report]")
print("  \\\\[")
print(f"    P(X=0) = \\\\frac{{\\\\binom{{{K}}}{{0}} \\\\binom{{{N-K}}}{{{n}}}}}{{\\\\binom{{{N}}}{{{n}}}}} \\\\approx {p_zero:.4f}")
print("  \\\\]")

print(f"\n{separator}")
print("  CONCLUSION: The probability P(X=0) = {:.6f} ({:.4f}%) is non-negligible.".format(p_zero, 100.0 * p_zero))
print("  Random splitting WITHOUT stratification is statistically unsound")
print("  for imbalanced datasets with a low positive rate such as 2%.")
print(separator)
