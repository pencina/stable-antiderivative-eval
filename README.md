# Stable Evaluation of Antiderivatives of ∫ xᵐ (ax+b)ⁿ dx

Reproducibility code for the JSC manuscript (`Main_JSC_v1.tex`). All reference
values are computed in 50-digit arithmetic (mpmath) so that reference error
never contaminates the measured float64 errors.

## Setup

On a modern Linux distribution the system Python is externally managed
(PEP 668), so install the dependencies into a local virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # numpy, mpmath, matplotlib, scipy, sympy
```

(Equivalently: `.venv/bin/pip install numpy mpmath matplotlib scipy sympy`.)

## Reproducing the results

```bash
.venv/bin/python make_figures.py                 # Figs. 1-3
.venv/bin/python reproduce_alpha_validation.py   # Table 1
.venv/bin/python reproduce_exp_example.py        # Remark 1
.venv/bin/python worst_case.py                   # Sec. 5 dense worst-case error
.venv/bin/python cas_output_forms.py             # Sec. 1 literal CAS outputs
.venv/bin/python sweep_cas_outputs.py            # Sec. 1 CAS-form sweep over (m,n)
```

- `make_figures.py` writes `fig_error_landscape.pdf`, `fig_anatomy_xstar.pdf`,
  and `fig_algorithm.pdf`, and prints a sanity line reporting the maximum
  Algorithm-1 relative error on the grid (~14 eps).
- `reproduce_alpha_validation.py` prints all seven rows of Table 1
  (predicted vs measured error amplification αₘ(n)).
- `reproduce_exp_example.py` prints the Remark 1 table (naive parts form vs
  the `expm1`-based rewrite of ∫ t eᵃᵗ dt).
- `worst_case.py` runs Algorithm 1 over a dense sweep of the series/F₁ handoff
  region and reports the Section 5 worst-case relative error (≈40 eps at
  n=−9/10, ≈280 eps at n=−99/100), with the predicted amplification
  ε·|F₁(0)|/I(x) for comparison.
- `cas_output_forms.py` prints the literal antiderivatives current CAS return
  (SymPy `integrate`/`manualintegrate`; Mathematica documented from a manual
  July 2026 check) and measures their large-x stability against F₁ and the
  two-term parts form F₂, backing the Section 1 paragraph "Where the two
  representations arise in practice". Imports `sympy`.
- `sweep_cas_outputs.py` checks that SymPy's output stays in the large-x-stable
  common-factor structure across the family (m ∈ {1,2,3}, exponents −99/100 to
  3/2; median errors 0.5–2.3 eps at large x), backing the same paragraph's
  claim that the α penalty attaches to the parts *representation*, not to
  present-day CAS front-ends. Imports `sympy`.
- `verification_checks.py` runs additional numerical sanity checks. CHECK 11
  is the superseded closed-form-only selector; CHECK 12 is the Algorithm 1
  hybrid (its before/after counterpart). CHECK 9 also imports `scipy`.

## Pinned versions

Exact versions used to produce the published figures are pinned in
`requirements.txt`. The float64 error figures are stable across numpy/mpmath
releases; the pins simply remove any ambiguity for future readers.

- numpy 2.5.1
- mpmath 1.4.1
- matplotlib 3.11.0
- scipy 1.18.0
- sympy 1.14.0
