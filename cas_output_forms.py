# cas_output_forms.py -- documents what CAS actually return for the paper's
# integrals, and measures the large-x stability of those literal outputs
# against F1 and the two-term parts form F2.  Supports the introduction's
# "Where the two representations arise in practice" paragraph.
#
# Wolfram|Alpha / Mathematica output (verified manually, July 2026):
#   Integrate[x Sqrt[a x + b], x]
#     = 2 Sqrt[b + a x] (-2 b^2 + a b x + 3 a^2 x^2) / (15 a^2)
#   (quadratic factor = (ax+b)(3ax-2b): cancellation root IMPLICIT)
#
# Requires: sympy, numpy, mpmath.

import sympy as sp
import numpy as np
from mpmath import mp, mpf, quad as mpquad

mp.dps = 50
x = sp.symbols('x', positive=True)

print("=== Literal SymPy outputs ===")
from sympy.integrals.manualintegrate import manualintegrate
for nn, tag in [(sp.Rational(1,2), 'n=1/2 '), (sp.Rational(-9,10), 'n=-9/10')]:
    print(f"integrate,       {tag}: {sp.integrate(x*(3*x+1)**nn, x)}")
    print(f"manualintegrate, {tag}: {manualintegrate(x*(3*x+1)**nn, x)}")

print("\n=== Large-x stability of literal outputs, n=-9/10, (a,b)=(3,1) ===")
a, b, n = 3.0, 1.0, -0.9
A, B, N = mpf(3), mpf(1), mpf(-9)/10
def refI(xx): return mpquad(lambda t: t*(A*t+B)**N, [0, mpf(xx)])
def F1(xx):
    u = a*xx+b
    return u**(n+2)/(a*a*(n+2)) - b*u**(n+1)/(a*a*(n+1))
def F2(xx):  # two-term parts form (pays alpha)
    u = a*xx+b
    return xx*u**(n+1)/(a*(n+1)) - u**(n+2)/(a*a*(n+1)*(n+2))
def F_sympy_literal(xx):  # LITERAL SymPy output (common-factor sum)
    return 10.0*xx*(3.0*xx+1.0)**0.1/33.0 - 100.0*(3.0*xx+1.0)**0.1/99.0
def F_sympy_simplified(xx):  # after simplify(): factored, root explicit
    return 10.0*(3.0*xx-10.0)*(3.0*xx+1.0)**0.1/99.0

r_lit, r_fact, r_two = [], [], []
for xx in np.logspace(2, 5, 20):
    r = refI(xx)
    e1 = abs((mpf(F1(xx)-F1(0)) - r)/r)
    el = abs((mpf(F_sympy_literal(xx)-F_sympy_literal(0)) - r)/r)
    ef = abs((mpf(F_sympy_simplified(xx)-F_sympy_simplified(0)) - r)/r)
    e2 = abs((mpf(F2(xx)-F2(0)) - r)/r)
    if e1 > 0:
        r_lit.append(float(el/e1)); r_fact.append(float(ef/e1))
        r_two.append(float(e2/e1))
print(f"median error ratio (SymPy literal sum)/F1 : {np.median(r_lit):.2f}"
      "   <- root implicit, large-x stable")
print(f"median error ratio (SymPy simplified)/F1  : {np.median(r_fact):.2f}"
      "   <- root explicit, large-x stable")
print(f"median error ratio (two-term parts F2)/F1 : {np.median(r_two):.2f}"
      "   <- pays alpha_1(-9/10)=11")
