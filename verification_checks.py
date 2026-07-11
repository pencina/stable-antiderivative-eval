# verification_checks.py -- AUDIT SUITE (checks 5-11)
#
# This script documents the numerical audit that shaped Sections 3-5 of the
# manuscript.  It is NOT a reproduction script for the paper's final claims
# (those are reproduce_alpha_validation.py, reproduce_exp_example.py, and
# make_figures.py).  Two checks intentionally exhibit FAILURE modes:
#
#   CHECK 9  shows that a float64 reference (scipy.quad, error ~1e-13)
#            masks the ~1e-15 effect entirely -- this is WHY the paper's
#            snippet uses a 50-digit mpmath reference.  The scipy-path
#            ratio of ~1.0 is the pitfall, not the result.
#
#   CHECK 11 tests the SUPERSEDED pre-correction algorithm (F2 for small x,
#            extended-precision band at x*, F1 for large x).  Its max error
#            of ~1e-4 at small x is what motivated replacing it with the
#            series+F1 hybrid of Algorithm 1 in the paper, whose verified
#            error is ~14-20 eps (see make_figures.py sanity output).
#
import numpy as np
from mpmath import mp, mpf, quad as mpquad
mp.dps = 50
a, b, n = 3.0, 1.0, -0.9
A, B, N = mpf(3), mpf(1), mpf(-9)/10

def refI(x): return mpquad(lambda t: t*(A*t+B)**N, [0, mpf(x)])
def F1(x):
    u=a*x+b; return u**(n+2)/(a*a*(n+2)) - b*u**(n+1)/(a*a*(n+1))
def F2(x):
    u=a*x+b; return x*u**(n+1)/(a*(n+1)) - u**(n+2)/(a*a*(n+1)*(n+2))

xstar = b/(a*(n+1))
print(f"x* = {xstar:.6f}")
print("\nCHECK 5: behavior near x* -- does the DEFINITE INTEGRAL error spike?")
print(f"{'x':>12} {'relerr F1':>12} {'relerr F2':>12}")
for d in [0.5, 0.1, 1e-2, 1e-4, 1e-6, 1e-8, 0.0, -1e-4, -0.1]:
    x = xstar*(1+d)
    r = refI(x)
    e1 = abs((mpf(F1(x)-F1(0)) - r)/r)
    e2 = abs((mpf(F2(x)-F2(0)) - r)/r)
    print(f"{x:12.6f} {float(e1):12.2e} {float(e2):12.2e}   (delta={d:g})")

print("\nCHECK 6: relative error of the ANTIDERIVATIVE VALUE F(x) itself near x*")
def refF(x):
    # exact antiderivative value (F1 form in high precision)
    U = A*mpf(x)+B
    return U**(N+2)/(A*A*(N+2)) - B*U**(N+1)/(A*A*(N+1))
for d in [1e-2, 1e-4, 1e-6, 1e-8]:
    x = xstar*(1+d)
    rF = refF(x)
    e1 = abs((mpf(F1(x)) - rF)/rF)
    e2 = abs((mpf(F2(x)) - rF)/rF)
    print(f"delta={d:8.0e}:  F1 expr relerr={float(e1):9.2e}   F2 expr relerr={float(e2):9.2e}")

print("\nCHECK 7: small-x regime for definite integral I=F(x)-F(0): which form wins?")
for x in [1e-8, 1e-6, 1e-4, 1e-2]:
    r = refI(x)
    e1 = abs((mpf(F1(x)-F1(0)) - r)/r)
    e2 = abs((mpf(F2(x)-F2(0)) - r)/r)
    print(f"x={x:8.0e}: F1={float(e1):9.2e}  F2={float(e2):9.2e}")

print("\nCHECK 8: missing table entry m=1, n=1/2 (I fabricated 1.5 -- measure it)")
n2 = 0.5; N2 = mpf(1)/2
def F1h(x):
    u=a*x+b; return u**(n2+2)/(a*a*(n2+2)) - b*u**(n2+1)/(a*a*(n2+1))
def F2h(x):
    u=a*x+b; return x*u**(n2+1)/(a*(n2+1)) - u**(n2+2)/(a*a*(n2+1)*(n2+2))
rr=[]
for x in np.logspace(2,5,25):
    r = mpquad(lambda t: t*(A*t+B)**N2, [0, mpf(x)])
    e1 = abs((mpf(F1h(x)-F1h(0)) - r)/r); e2 = abs((mpf(F2h(x)-F2h(0)) - r)/r)
    if e1>0: rr.append(float(e2/e1))
print(f"m=1, n=1/2: predicted alpha=5/3={5/3:.2f}, measured median={np.median(rr):.2f}")

print("\nCHECK 9 [PITFALL DEMO]: float64 scipy reference masks the effect;")
print("only the 50-digit line below is meaningful (the paper snippet uses mpmath)")
from scipy.integrate import quad as squad
ref,_ = squad(lambda t: t*(a*t+b)**n, 0, 1e4)
o1 = abs((F1(1e4)-F1(0)-ref)/ref); o2 = abs((F2(1e4)-F2(0)-ref)/ref)
print(f"snippet prints: {o1:.3e} and {o2:.3e}, ratio={o2/o1:.1f}")
# also against 50-digit reference
r50 = refI(1e4)
print(f"vs 50-digit ref: F1={float(abs((mpf(F1(1e4)-F1(0))-r50)/r50)):.3e}  F2={float(abs((mpf(F2(1e4)-F2(0))-r50)/r50)):.3e}")

print("\nCHECK 10: (a,b) independence spot check, m=1 n=-0.9, (a,b)=(7,5) and (0.5,2)")
for (aa,bb) in [(7.0,5.0),(0.5,2.0)]:
    AA,BB = mpf(aa),mpf(bb)
    def F1x(x): u=aa*x+bb; return u**(n+2)/(aa*aa*(n+2)) - bb*u**(n+1)/(aa*aa*(n+1))
    def F2x(x): u=aa*x+bb; return x*u**(n+1)/(aa*(n+1)) - u**(n+2)/(aa*aa*(n+1)*(n+2))
    rr=[]
    for x in np.logspace(2,5,15):
        r = mpquad(lambda t: t*(AA*t+BB)**N, [0, mpf(x)])
        e1=abs((mpf(F1x(x)-F1x(0))-r)/r); e2=abs((mpf(F2x(x)-F2x(0))-r)/r)
        if e1>0: rr.append(float(e2/e1))
    print(f"(a,b)=({aa},{bb}): measured median ratio={np.median(rr):.1f}  (predicted 11)")

print("\nCHECK 11 [SUPERSEDED ALGORITHM]: pre-correction closed-form-only selector;")
print("its small-x failure (~1e-4) motivated the series+F1 hybrid of Algorithm 1")
def alg(x):
    xs = b/(a*(n+1)); dtol = 10*np.sqrt(2.2e-16)
    if x < xs/2: return F2(x)-F2(0)
    elif abs(x-xs) < dtol*max(xs,1.0): return float(refI(x))  # ext-precision fallback
    else: return F1(x)-F1(0)
errs=[]
for x in np.logspace(-6,6,60):
    r = refI(x); v = alg(x)
    errs.append(float(abs((mpf(v)-r)/r)))
print(f"max relerr = {max(errs):.2e}  ({max(errs)/2.2e-16:.1f} eps_mach)")


print("\nCHECK 12 [ALGORITHM 1 OF THE PAPER]: corrected series+F1 hybrid,")
print("max relerr over x in [1e-6,1e6] -- the before/after counterpart of CHECK 11")
EPS = 2.220446049250313e-16
def run_alg1(nn):
    NN = mpf(repr(nn)) if False else mpf(nn)  # binary double value of n (see provenance note in Sec. 4)
    def F1n(x):
        u=a*x+b; return u**(nn+2)/(a*a*(nn+2)) - b*u**(nn+1)/(a*a*(nn+1))
    def ser(x, kmax=400):
        sacc = b**nn * x*x/2.0; c=1.0
        for k in range(1,kmax):
            c *= (nn-(k-1))/k
            t = c*a**k*b**(nn-k)*x**(k+2)/(k+2)
            sacc += t
            if abs(t) < 0.25*EPS*abs(sacc): break
        return sacc
    def alg1(x):
        return ser(x) if (b>0 and a*x<=0.9*b) else F1n(x)-F1n(0)
    errs=[]
    for x in np.logspace(-6,6,60):
        r = mpquad(lambda t: t*(A*t+B)**NN,[0,mpf(x)])
        errs.append(float(abs((mpf(alg1(x))-r)/r)))
    return max(errs)
for nn in [-0.9, -0.99]:
    mx = run_alg1(nn)
    print(f"n={nn}: max relerr = {mx:.2e}  ({mx/EPS:.0f} eps)"
          + ("   <- paper Sec. 5: worst ~40 eps (dense sweep); coarse grids 14-39" if nn==-0.9 else "   <- paper Sec. 5: worst ~280 eps (dense sweep), O(eps/|n+1|)"))
