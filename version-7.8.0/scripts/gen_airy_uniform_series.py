#!/usr/bin/env python3
"""Generate the constants in `special_functions/src/airy_uniform.rs`.

NOT part of the build, and NOT a dependency: this is the working that
produced the numbers, kept so they can be re-derived and argued with
rather than taken on trust. It needs mpmath and sympy; the crate itself
needs neither, and still has no dependencies at all.

The constants it emits are verified inside the crate, against the closed
forms, by `airy_uniform::tests::the_generated_series_match_the_closed_forms`
— so this file is the provenance, not the authority.

    python3 scripts/gen_airy_uniform_series.py
"""
import mpmath as mp, sympy as sp
mp.mp.dps = 70

# --- Debye polynomials U_k(p), exact via sympy -----------------------
p = sp.symbols('p')
U = [sp.Integer(1)]
for k in range(9):
    t = sp.Rational(1,2)*p**2*(1-p**2)*sp.diff(U[k], p)
    t += sp.Rational(1,8)*sp.integrate((1-5*sp.Symbol('t')**2)*U[k].subs(p, sp.Symbol('t')), (sp.Symbol('t'), 0, p))
    U.append(sp.expand(t))
print("U1 =", U[1], " U2 =", U[2])

Uf = [sp.lambdify(p, u, 'mpmath') for u in U]

# --- lambda_j, mu_j (DLMF 10.20.12/13) -------------------------------
def lam(j):
    if j == 0: return sp.Integer(1)
    num = sp.Integer(1)
    for m in range(2*j+1, 6*j, 2):
        num *= m
    return sp.Rational(num, sp.factorial(j)*144**j)
def mu(j):
    if j == 0: return sp.Integer(1)
    return -sp.Rational(6*j+1, 6*j-1)*lam(j)
LAM = [mp.mpf(sp.nsimplify(lam(j)).evalf(70)) for j in range(9)]
MU  = [mp.mpf(sp.nsimplify(mu(j)).evalf(70))  for j in range(9)]
print("lam:", [mp.nstr(v,8) for v in LAM[:4]])
print("mu :", [mp.nstr(v,8) for v in MU[:4]])

# --- zeta(x) ---------------------------------------------------------
def zeta(x):
    x = mp.mpf(x)
    if x < 1:
        s = mp.sqrt(1-x*x)
        f = mp.log((1+s)/x) - s          # = (2/3) zeta^{3/2}
        return (mp.mpf(3)/2*f)**(mp.mpf(2)/3)
    elif x > 1:
        s = mp.sqrt(x*x-1)
        f = s - mp.acos(1/x)             # = (2/3)(-zeta)^{3/2}
        return -(mp.mpf(3)/2*f)**(mp.mpf(2)/3)
    else:
        return mp.mpf(0)

# --- A_k(zeta), B_k(zeta) as functions of x --------------------------
def AB(k, x):
    """(A_k, B_k) at this x, via DLMF 10.20.10/10.20.11."""
    x = mp.mpf(x)
    z = zeta(x)
    pv = 1/mp.sqrt(mp.mpc(1-x*x))        # complex: imaginary for x>1
    zp = lambda e: mp.power(mp.mpc(z), e) # principal branch
    A = mp.mpc(0)
    for j in range(0, 2*k+1):
        A += MU[j]*zp(mp.mpf(-3*j)/2)*Uf[2*k-j](pv)
    B = mp.mpc(0)
    for j in range(0, 2*k+2):
        B += LAM[j]*zp(mp.mpf(-3*j)/2)*Uf[2*k+1-j](pv)
    B *= -zp(mp.mpf(-1)/2)
    return A, B


import mpmath as mp
mp.mp.dps = 70
N = 18          # Taylor terms
R = mp.mpf('0.35')   # fit half-width in w = 1 - x

def taylor_of(f, n=N, r=R):
    """Interpolate f(w) on n Chebyshev points of [-r, r] and return the
    coefficients of the interpolating polynomial (== Taylor coefficients
    to well past f64 precision, since f is analytic there)."""
    ws = [r*mp.cos(mp.pi*(2*i+1)/(2*n)) for i in range(n)]
    V = mp.matrix(n, n)
    y = mp.matrix(n, 1)
    for i, w in enumerate(ws):
        for j in range(n):
            V[i, j] = w**j
        y[i] = f(w)
    return mp.lu_solve(V, y)

zf  = lambda w: zeta(1-w)/w if w != 0 else mp.mpf(2)**(mp.mpf(1)/3)
b0f = lambda w: AB(0, 1-w)[1].real
a1f = lambda w: AB(1, 1-w)[0].real
b1f = lambda w: AB(1, 1-w)[1].real
a2f = lambda w: AB(2, 1-w)[0].real
b2f = lambda w: AB(2, 1-w)[1].real

names = [('ZETA_OVER_W', zf), ('B0_W', b0f), ('A1_W', a1f), ('B1_W', b1f),
         ('A2_W', a2f), ('B2_W', b2f)]
out = []
for nm, f in names:
    c = taylor_of(f)
    # verify: max |series - exact| over |w| <= 0.25
    worst = mp.mpf(0)
    for k in range(-25, 26):
        w = mp.mpf(k)/100
        if w == 0: continue
        s = sum(c[j]*w**j for j in range(N))
        e = abs(s - f(w))/max(abs(f(w)), mp.mpf('1e-30'))
        worst = max(worst, e)
    print(f"{nm}: worst relative residual over |w|<=0.25 = {mp.nstr(worst,4)}")
    body = ",\n    ".join(mp.nstr(c[j], 20, strip_zeros=False) for j in range(N))
    out.append(f"const {nm}: [f64; {N}] = [\n    {body},\n];")
open('airy_uniform_series.rs','w').write("\n\n".join(out))
print("\n--- first few coefficients ---")
for nm, f in names:
    c = taylor_of(f)
    print(nm, [mp.nstr(c[j], 12) for j in range(4)])
