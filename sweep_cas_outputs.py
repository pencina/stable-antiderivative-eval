# sweep_cas_outputs.py -- verifies that current SymPy output stays in the
# large-x-stable common-factor structure across the family (m, n): supports
# the introduction claim that the alpha penalty attaches to the parts
# REPRESENTATION (reduction formulas, hand derivations), not to present-day
# CAS front-ends.  Requires: sympy, numpy, mpmath.
import sympy as sp
import numpy as np
from mpmath import mp, mpf, quad as mpquad
mp.dps = 50
x = sp.symbols('x', positive=True)
a_, b_ = 3, 1
A, B = mpf(3), mpf(1)

def stable_ref_form(m, n):
    # high-precision antiderivative via substitution (F1-type), exact
    def F(xx):
        U = A*mpf(xx)+B
        s = mpf(0)
        # int x^m u^n dx = a^{-(m+1)} int (u-b)^m u^n du, binomial expansion
        from mpmath import binomial
        for k in range(m+1):
            c = binomial(m, k) * (-B)**(m-k)
            s += c * U**(n+k+1)/(n+k+1)
        return s / A**(m+1)
    return F

def test(m, n_sym, tag):
    n = float(n_sym)
    expr = sp.integrate(x**m * (a_*x+b_)**n_sym, x)
    f = sp.lambdify(x, expr, 'numpy')     # literal float64 evaluation
    N = mpf(n)
    refF = stable_ref_form(m, N)
    def refI(xx): return mpquad(lambda t: t**m*(A*t+B)**N, [0, mpf(xx)])
    ratios = []
    for xx in np.logspace(2, 5, 12):
        r = refI(xx)
        e_cas = abs((mpf(float(f(xx)) - float(f(1e-30))) - r)/r)  # F(x)-F(0)
        ratios.append(float(e_cas / (2.2e-16)))
    med = np.median(ratios)
    # structural check: does the expression contain a subtraction of terms with
    # the same leading power of x?
    print(f"m={m}, n={tag}:  median relerr = {med:.1f} eps   output: {sp.simplify(expr)}")

for m, nn, tag in [(1, sp.Rational(-9,10), '-9/10'),
                   (2, sp.Rational(-9,10), '-9/10'),
                   (3, sp.Rational(-9,10), '-9/10'),
                   (2, sp.Rational(-99,100), '-99/100'),
                   (1, sp.Rational(-1,3), '-1/3'),
                   (2, sp.Rational(1,2), '1/2'),
                   (1, sp.Rational(3,2), '3/2'),
                   (2, sp.Rational(3,2), '3/2'),
                   (2, sp.Rational(-5,3), '-5/3')]:
    try:
        test(m, nn, tag)
    except Exception as e:
        print(f"m={m}, n={tag}: FAILED ({e})")
