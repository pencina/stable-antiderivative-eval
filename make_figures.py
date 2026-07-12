# make_figures.py
# Generates the figures for "Stable Evaluation of Antiderivatives of
# int x^m (ax+b)^n dx".  All reference values are computed in 50-digit
# arithmetic (mpmath) so that reference error never contaminates the
# measured float64 errors.  Requires: numpy, matplotlib, mpmath.
#
#   Fig. 1  fig_error_landscape.pdf   relerr of F1, F2 (definite integral)
#                                     vs x; band separation ~ alpha_1(n)=11
#   Fig. 2  fig_anatomy_xstar.pdf     top: |(n+1)ax-b|; bottom: expression
#                                     vs definite-integral error near x*
#   Fig. 3  fig_algorithm.pdf         (optional, not yet referenced in the
#                                     paper) corrected algorithm error
#                                     profile across twelve decades

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpmath import mp, mpf, quad as mpquad

mp.dps = 50
EPS = 2.220446049250313e-16

# ----------------------------------------------------------------------
# problem setup: case study m=1, n=-9/10, (a,b)=(3,1)
a, b, n = 3.0, 1.0, -0.9
A, B, N = mpf(3), mpf(1), mpf(-9) / 10
XSTAR = b / (a * (n + 1.0))                      # = 10/3

def refI(x):
    """50-digit reference for the definite integral I(x)."""
    return mpquad(lambda t: t * (A * t + B) ** N, [0, mpf(x)])

def refF(x):
    """50-digit reference for the antiderivative value F(x) (F1 form)."""
    U = A * mpf(x) + B
    return U ** (N + 2) / (A * A * (N + 2)) - B * U ** (N + 1) / (A * A * (N + 1))

def F1(x):
    u = a * x + b
    return u ** (n + 2) / (a * a * (n + 2)) - b * u ** (n + 1) / (a * a * (n + 1))

def F2(x):
    u = a * x + b
    return x * u ** (n + 1) / (a * (n + 1)) - u ** (n + 2) / (a * a * (n + 1) * (n + 2))

def I_series(x, kmax=400):
    """Binomial series (eq. 12 in the paper), |ax/b| < 1."""
    s = b ** n * x * x / 2.0
    c = 1.0
    for k in range(1, kmax):
        c *= (n - (k - 1)) / k
        t = c * a ** k * b ** (n - k) * x ** (k + 2) / (k + 2)
        s += t
        if abs(t) < 0.25 * EPS * abs(s):
            break
    return s

def relerr(v, r):
    return float(abs((mpf(v) - r) / r))

# ----------------------------------------------------------------------
# global style: sized for elsarticle single column (~5.3 in text width)
plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "figure.dpi": 150,
})
C1, C2, C3 = "#1a5fb4", "#c01c28", "#26a269"   # F1 blue, F2 red, aux green

# ======================================================================
# Figure 1: error landscape of the definite integral
# ======================================================================
print("Figure 1 ...")
xs = np.logspace(-2, 6, 220)
e1 = np.empty_like(xs)
e2 = np.empty_like(xs)
for i, x in enumerate(xs):
    r = refI(x)
    e1[i] = relerr(F1(x) - F1(0), r)
    e2[i] = relerr(F2(x) - F2(0), r)

fig, ax = plt.subplots(figsize=(5.3, 3.1))
ax.loglog(xs, e2, ".", ms=2.5, color=C2, alpha=0.75,
          label=r"$F_2$ (parts form)"
ax.loglog(xs, e1, ".", ms=2.5, color=C1, alpha=0.75, label=r"$F_1$")
ax.axhline(EPS, color="0.55", lw=0.6, ls=":")
ax.text(2e-2, EPS * 1.35, r"$\epsilon_{\rm mach}$", color="0.35", fontsize=8)
ax.axhline(11 * EPS, color="0.55", lw=0.6, ls="--")
ax.text(2e-2, 11 * EPS * 1.35, r"$\alpha_1(-9/10)\,\epsilon_{\rm mach}=11\,\epsilon_{\rm mach}$",
        color="0.35", fontsize=8)
ax.axvline(XSTAR, color=C3, lw=0.7, ls="-.")
ax.text(XSTAR * 1.25, 3e-13, r"$x^{\star}=\frac{b}{a(n+1)}$",
        color=C3, fontsize=8)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"relative error of $F(x)-F(0)$")
ax.set_ylim(3e-17, 2e-12)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("fig_error_landscape.pdf")
plt.close(fig)

# ======================================================================
# Figure 2: anatomy of the critical point
# ======================================================================
print("Figure 2 ...")
deltas = np.logspace(-9, -0.5, 34)

exp1 = np.empty_like(deltas)   # expression relerr of F1(x)
exp2 = np.empty_like(deltas)   # expression relerr of F2(x)
di1  = np.empty_like(deltas)   # definite-integral relerr, F1
di2  = np.empty_like(deltas)   # definite-integral relerr, F2
for i, d in enumerate(deltas):
    x  = XSTAR * (1.0 + d)
    rF = refF(x)
    rI = refI(x)
    exp1[i] = relerr(F1(x), rF)
    exp2[i] = relerr(F2(x), rF)
    di1[i]  = relerr(F1(x) - F1(0), rI)
    di2[i]  = relerr(F2(x) - F2(0), rI)

fig, (axt, axb) = plt.subplots(
    2, 1, figsize=(5.3, 4.4), height_ratios=[1, 1.6], sharex=False)

# --- top panel: the vanishing factor ---------------------------------
xg = np.linspace(0.0, 8.0, 400)
axt.plot(xg, np.abs((n + 1) * a * xg - b), color="0.2", lw=1.0)
axt.axvline(XSTAR, color=C3, lw=0.8, ls="-.")
axt.text(XSTAR + 0.12, 0.85, r"$x^{\star}=b/[a(n+1)]$", color=C3, fontsize=8)
axt.set_xlabel(r"$x$")
axt.set_ylabel(r"$|(n+1)ax-b|$")
axt.set_xlim(0, 8)
axt.set_ylim(bottom=0)

# --- bottom panel: expression vs definite-integral error --------------
axb.loglog(deltas, exp2, "s-", ms=3, color=C2, lw=0.9,
           label=r"$F_2(x)$  (expression value)")
axb.loglog(deltas, exp1, "o-", ms=3, color=C1, lw=0.9,
           label=r"$F_1(x)$  (expression value)")
axb.loglog(deltas, di2, "s--", ms=3, color=C2, lw=0.9, mfc="none",
           label=r"$F_2(x)-F_2(0)$  (definite integral)")
axb.loglog(deltas, di1, "o--", ms=3, color=C1, lw=0.9, mfc="none",
           label=r"$F_1(x)-F_1(0)$  (definite integral)")
guide = EPS / deltas
axb.loglog(deltas, guide, ":", color="0.5", lw=0.8)
axb.text(2.5e-8, EPS / 2.5e-8 * 2.5, r"$\epsilon_{\rm mach}/|\delta|$",
         color="0.35", fontsize=8, rotation=-24)
axb.set_xlabel(r"$|\delta|$,   where $x=x^{\star}(1+\delta)$")
axb.set_ylabel("relative error")
axb.set_ylim(5e-18, 5e-6)
axb.legend(loc="upper right", fontsize=7.2)
fig.tight_layout()
fig.savefig("fig_anatomy_xstar.pdf")
plt.close(fig)

# ======================================================================
# Figure 3 (optional): corrected algorithm across twelve decades
# ======================================================================
print("Figure 3 ...")
xs = np.logspace(-6, 6, 160)
e_alg = np.empty_like(xs)
e_f2  = np.empty_like(xs)
for i, x in enumerate(xs):
    r = refI(x)
    v = I_series(x) if (b > 0 and a * x <= 0.9 * b) else F1(x) - F1(0)
    e_alg[i] = relerr(v, r)
    e_f2[i]  = relerr(F2(x) - F2(0), r)

fig, ax = plt.subplots(figsize=(5.3, 3.0))
ax.loglog(xs, e_f2, ".", ms=2.5, color=C2, alpha=0.7,
          label=r"CAS default: $F_2(x)-F_2(0)$")
ax.loglog(xs, e_alg, ".", ms=2.5, color=C1, alpha=0.9,
          label="Algorithm 1 (series + $F_1$)")
ax.axhline(EPS, color="0.55", lw=0.6, ls=":")
ax.text(2e-6, EPS * 1.4, r"$\epsilon_{\rm mach}$", color="0.35", fontsize=8)
ax.axvline(0.9 * b / a, color=C3, lw=0.7, ls="-.")
ax.text(0.9 * b / a * 1.3, 1e-6, "series/$F_1$\nhandoff", color=C3, fontsize=7.5)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"relative error of $I(x)$")
ax.set_ylim(3e-17, 3e0)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("fig_algorithm.pdf")
plt.close(fig)

print("done: fig_error_landscape.pdf, fig_anatomy_xstar.pdf, fig_algorithm.pdf")
print(f"sanity: max Algorithm-1 error on grid = {e_alg.max():.2e} "
      f"({e_alg.max()/EPS:.1f} eps)")
