import numpy as np
from mpmath import mp, mpf, quad as mpquad
mp.dps = 50
a, b = 3.0, 1.0
A, B = mpf(3), mpf(1)
EPS = 2.220446049250313e-16

def worst(nn, xs):
    NN = mpf(nn)
    def F1n(x):
        u=a*x+b; return u**(nn+2)/(a*a*(nn+2)) - b*u**(nn+1)/(a*a*(nn+1))
    def ser(x, kmax=400):
        s = b**nn * x*x/2.0; c=1.0
        for k in range(1,kmax):
            c *= (nn-(k-1))/k
            t = c*a**k*b**(nn-k)*x**(k+2)/(k+2)
            s += t
            if abs(t) < 0.25*EPS*abs(s): break
        return s
    def alg1(x):
        return ser(x) if (b>0 and a*x<=0.9*b) else F1n(x)-F1n(0)
    mx, mxx = 0.0, None
    for x in xs:
        r = mpquad(lambda t: t*(A*t+B)**NN,[0,mpf(x)])
        e = float(abs((mpf(alg1(x))-r)/r))
        if e > mx: mx, mxx = e, x
    return mx, mxx

# dense sweep concentrated on the handoff region [0.2, 1.2] + sparse elsewhere
xs_dense  = np.linspace(0.20, 1.20, 400)
xs_sparse = np.logspace(-6, 6, 60)
for nn in [-0.9, -0.99]:
    m1, x1 = worst(nn, xs_dense)
    m2, x2 = worst(nn, xs_sparse)
    m = max(m1, m2); xloc = x1 if m1>=m2 else x2
    # theoretical amplification at handoff: |F1(0)| / I(x_handoff)
    NN = mpf(nn)
    F10 = abs(b**(nn+2)/(a*a*(nn+2)) - b*b**(nn+1)/(a*a*(nn+1)))
    Ih = mpquad(lambda t: t*(A*t+B)**NN,[0,mpf(0.3)])
    print(f"n={nn}: worst = {m:.2e} ({m/EPS:.0f} eps) at x={xloc:.4f};"
          f"  |F1(0)|/I(0.3) = {float(F10/Ih):.1f}")
