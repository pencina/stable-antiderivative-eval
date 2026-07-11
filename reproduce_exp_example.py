# Reproduces Remark 1 of the manuscript: the CAS-returned form of
# int_0^x t e^{at} dt loses all significant digits as x->0, while the
# expm1-based rewrite is uniformly accurate. Requires: numpy, mpmath.
import numpy as np
from mpmath import mp, mpf, e as mpe, exp as mpexp

mp.dps = 50
a = 3.0
A = mpf(3)

# I(x) = int_0^x t e^{at} dt = [e^{ax}(ax-1) + 1] / a^2   (exact)
def ref(x):
    X = mpf(x)
    return (mpexp(A*X)*(A*X-1) + 1)/A**2

# Naive parts form in float64:  F(x) = e^{ax}(ax-1)/a^2 ;  I = F(x)-F(0)
def I_naive(x):
    F  = np.exp(a*x)*(a*x-1.0)/a**2
    F0 = np.exp(0.0)*(0.0-1.0)/a**2     # = -1/a^2
    return F - F0

# Stable rewrite using expm1:  I = [a x e^{ax} - expm1(a x)] / a^2
def I_stable(x):
    return (a*x*np.exp(a*x) - np.expm1(a*x))/a**2

print(f"{'x':>10} {'relerr naive':>14} {'relerr stable':>14} {'ratio':>10}")
for x in [1e-8, 1e-6, 1e-4, 1e-2, 1/3.0, 1.0, 10.0]:
    r = ref(x)
    en = abs((mpf(I_naive(x))  - r)/r)
    es = abs((mpf(I_stable(x)) - r)/r)
    ratio = float(en/es) if es > 0 else float('inf')
    print(f"{x:10.1e} {float(en):14.2e} {float(es):14.2e} {ratio:10.1f}")

# near the critical point x* = 1/a  (where (ax-1) vanishes)
print("\nNear x* = 1/a (cancellation inside the factor (ax-1)):")
for d in [1e-2, 1e-4, 1e-6, 1e-8]:
    x = (1.0 + d)/a
    r = ref(x)
    en = abs((mpf(I_naive(x)) - r)/r)
    es = abs((mpf(I_stable(x)) - r)/r)
    print(f"delta={d:8.0e}: naive={float(en):9.2e}  stable={float(es):9.2e}")
