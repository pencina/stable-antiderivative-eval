# Reproduces Table 1 of the manuscript (all seven rows): predicted vs
# measured error amplification alpha_m(n) = |n+m+1|/|n+1| of the
# integration-by-parts form F2 relative to the substitution form F1,
# for I_m(x) = int_0^x t^m (a t + b)^n dt, (a,b)=(3,1), median over
# x in [1e2, 1e5].  Reference values use 50-digit arithmetic (mpmath)
# so that reference error does not contaminate the measured ratios.
# Requires: numpy, mpmath.

import numpy as np
from mpmath import mp, mpf, quad as mpquad

mp.dps = 50
a, b = 3.0, 1.0
A, B = mpf(3), mpf(1)

def forms(m, n):
    if m == 1:
        def F1(x):
            u = a*x + b
            return u**(n+2)/(a*a*(n+2)) - b*u**(n+1)/(a*a*(n+1))
        def F2(x):
            u = a*x + b
            return x*u**(n+1)/(a*(n+1)) - u**(n+2)/(a*a*(n+1)*(n+2))
    else:  # m == 2; F2 = one-step parts with the inner integral done stably
        def F1(x):
            u = a*x + b
            return (u**(n+3)/(n+3) - 2*b*u**(n+2)/(n+2)
                    + b*b*u**(n+1)/(n+1))/a**3
        def F2(x):
            u = a*x + b
            J  = (u**(n+3)/(n+3) - b*u**(n+2)/(n+2))/a**2
            J0 = (b**(n+3)/(n+3) - b*b**(n+2)/(n+2))/a**2
            return x*x*u**(n+1)/(a*(n+1)) - 2.0/(a*(n+1))*(J - J0)
    return F1, F2

def measured_ratio(m, n, xs):
    F1, F2 = forms(m, n)
    N = mpf(n)  # binary double value of n: the SAME exponent the float64 forms use.
    # Using the exact decimal (mpf(repr(n))) instead would introduce a
    # ~1e-17 model mismatch that inflates the tiny measured errors.
    r = []
    for x in xs:
        ref = mpquad(lambda t: t**m * (A*t + B)**N, [0, mpf(x)])
        e1 = abs((mpf(F1(x) - F1(0.0)) - ref)/ref)
        e2 = abs((mpf(F2(x) - F2(0.0)) - ref)/ref)
        if e1 > 0:
            r.append(float(e2/e1))
    return np.median(r)

rows = [(1, 0.5), (1, -0.9), (1, -0.99),
        (2, 0.5), (2, -0.9), (2, -0.95), (2, -0.99)]
xs = np.logspace(2, 5, 25)

print(f"{'m':>2} {'n':>7} {'alpha predicted':>16} {'measured':>10} {'bits':>6}")
for m, n in rows:
    pred = abs(n + m + 1)/abs(n + 1)
    med = measured_ratio(m, n, xs)
    print(f"{m:>2} {n:>7} {pred:>16.2f} {med:>10.2f} {np.log2(pred):>6.2f}")
print("\nNote (dagger footnote of Table 1): for alpha close to 1 the")
print("penalty is masked by the 2-3 eps rounding of the power evaluations")
print("themselves; the predictor is informative when alpha >> 1.")
