"""Numerical checks behind the referee report on
'Noise-Tolerant GHZ-Based Quantum Contract Signing' (main.tex).
Run: python qcs_review_checks.py   (numpy, scipy)
Equation numbers follow the compiled order of main.tex (body 1-46).
"""
import itertools, math
import numpy as np
from scipy import stats
from scipy.special import gammaln, logsumexp
from scipy.optimize import linprog, brentq

I = np.eye(2); X = np.array([[0, 1], [1, 0]]); Z = np.diag([1, -1]); Y = np.array([[0, -1j], [1j, 0]])
k3 = lambda a, b, c: np.kron(np.kron(a, b), c)
ghz = np.zeros(8); ghz[[0, 7]] = 1 / np.sqrt(2)
ghzm = np.zeros(8); ghzm[0], ghzm[7] = 1 / np.sqrt(2), -1 / np.sqrt(2)

print("== Eq.(4): stabilizer relations under X^a Z^u, all 64 label choices")
P = lambda a, u: np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, u)
ok = True
for aA, aB, aC, uA, uB, uC in itertools.product([0, 1], repeat=6):
    psi = k3(P(aA, uA), P(aB, uB), P(aC, uC)) @ ghz
    ev = lambda O: np.real(psi.conj() @ O @ psi)
    ok &= np.isclose(ev(k3(X, X, X)), (-1) ** (uA ^ uB ^ uC))
    ok &= np.isclose(ev(k3(Z, Z, I)), (-1) ** (aA ^ aB)) and np.isclose(ev(k3(I, Z, Z)), (-1) ** (aB ^ aC))
    ok &= np.isclose(ev(k3(Z, I, Z)), (-1) ** (aA ^ aC))
print("  holds:", ok)

print("== de Sousa-Ramos state = H^{x3}|GHZ>:", np.allclose(k3(*(np.array([[1, 1], [1, -1]]) / np.sqrt(2),) * 3) @ ghz,
                                                   np.array([1, 0, 0, 1, 0, 1, 1, 0]) / 2))

print("== Mermin identity")
Mop = k3(X, X, X) - k3(X, Y, Y) - k3(Y, X, Y) - k3(Y, Y, X)
print("  M = XXX(I+ZZI+ZIZ+IZZ):", np.allclose(Mop, k3(X, X, X) @ (np.eye(8) + k3(Z, Z, I) + k3(Z, I, Z) + k3(I, Z, Z))))
print("  M = 4(|G+><G+| - |G-><G-|):", np.allclose(Mop, 4 * (np.outer(ghz, ghz) - np.outer(ghzm, ghzm))))

print("== A-carrier algebra: received words = c xor residual; c = u_A xor alpha from openings")
rng = np.random.default_rng(1); ok = True
for _ in range(5000):
    c, al, be, ga, p, q, r = rng.integers(0, 2, 7)
    uA, uB, uC, aA, aB, aC = c ^ al, be, ga, c ^ p, q, r
    ok &= ((uA ^ uB ^ uC) ^ al ^ be ^ ga) == c and ((aA ^ aC) ^ p ^ r) == c and (uA ^ al) == c
print("  holds:", ok)

print("== Mermin finite-size, Eqs.(19),(20),(36)-(38)")
eM = 1e-10; L = math.log(1 / eM)
dN = lambda N: 4 * math.sqrt(2 * L / N)
print("  N_M from Eq.(20), g=0.4:", 32 * L / 0.16, "-> ceil", math.ceil(32 * L / 0.16), "; delta_N(4606) =", round(dN(4606), 4))
print("  N_M below which M_L>=3.2 is impossible for any strategy:", round(32 * L / 0.8 ** 2, 1))
Kmin = lambda N, Mmin=3.2: math.ceil((4 + Mmin) * N / 8 + math.sqrt(N * L / 2))
def logsf(N, K, p):
    k = np.arange(K, N + 1)
    return logsumexp(gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1) + k * np.log(p) + (N - k) * np.log1p(-p))
print("  K_min(4606) =", Kmin(4606))
for lab, p in [("local p=3/4", .75), ("v=0.85", .925), ("v=0.90", .95), ("v=0.96", .98)]:
    ls = logsf(4606, Kmin(4606), p)
    print(f"  {lab:12s} P_cert = {math.exp(ls):.4g}   1-P = {stats.binom.cdf(Kmin(4606) - 1, 4606, p):.3g}")
print("  log10 P_cert^local vs N (Fig.1b lower panel; double underflow below ~-323.3):")
for N in [1152, 3000, 4606, 5000, 5400, 6000, 10000, 30000]:
    print(f"    N={N:6d}  {logsf(N, Kmin(N), .75) / math.log(10):8.1f}")
print("  Azuma bound, any memoryful local strategy, N=4606: log10 P <=",
      round(-4606 * (1.2 + dN(4606)) ** 2 / 32 / math.log(10), 1))
print("  N_M for completeness 1-1e-10 at g=0.4:", round(32 * (2 * math.sqrt(L)) ** 2 / 0.16))

print("== Kaniewski (PRL 117, 070402) tight Mermin extractability, gamma* = 2 sqrt 2")
for g in [3.2, 3.6]:
    print(f"  M={g}: Q >= {0.5 + 0.5 * (g - 2 * math.sqrt(2)) / (4 - 2 * math.sqrt(2)):.4f}")

print("== Serfling profile, Eqs.(22),(33),(34)")
xi = lambda m, n, e: math.sqrt((m + n) * (n + 1) / (2 * m * n ** 2) * math.log(1 / e))
m, n, eps = 16384, 16383, 2.5e-11
x1 = xi(m, n, eps); print(f"  per-stream n=16383: xi={x1:.5f} eU={0.02 + x1:.5f} ceil(n eU)={math.ceil(n * (0.02 + x1))}")
x2 = xi(m, 2 * n, eps); print(f"  pooled X n=32766:  xi={x2:.5f} eU={0.02 + x2:.5f} worst single stream={math.ceil(2 * n * (0.02 + x2))}")
print(f"  white-noise v=0.9 (e=0.05): ceil(n eU)={math.ceil(n * (0.05 + x1))};  max e_hat for t=1000: {1000 / n - x1:.4f}")
print("  N_alloc =", 4606 + 32768 + 4 * 16383 + 5000, "; BCH dim bound =", 16383 - 14 * 1000)

print("== Consistency of Mhat=3.6 with residual rates (GHZ-diagonal LP)")
idx = [(s, b) for s in '+-' for b in '0ABC']
def Mrange(e):
    c = np.array([4 if (s, b) == ('+', '0') else -4 if (s, b) == ('-', '0') else 0 for s, b in idx], float)
    A = [np.ones(8), [s == '-' for s, b in idx], [b in 'AB' for s, b in idx], [b in 'BC' for s, b in idx], [b in 'AC' for s, b in idx]]
    lo = linprog(c, A_eq=np.array(A, float), b_eq=[1, e, e, e, e], bounds=[(0, 1)] * 8).fun
    hi = -linprog(-c, A_eq=np.array(A, float), b_eq=[1, e, e, e, e], bounds=[(0, 1)] * 8).fun
    return lo, hi
print("  all rates 0.02 -> M in [%.3f, %.3f]" % Mrange(0.02))
print("  M=3.6 compatible with equal rates e in [%.4f, %.4f]" % (brentq(lambda e: Mrange(e)[0] - 3.6, 1e-4, .2),
                                                                 brentq(lambda e: Mrange(e)[1] - 3.6, 1e-4, .2)))

print("== Working-residual figure, Eqs.(39)-(42)")
e = 0.02; qX = e + (1 - 2 * e) * 0.04
for r in [800, 900, 1000]:
    print(f"  r={r}: P_raw = {stats.binom.cdf(r, n, qX) ** 2 * stats.binom.cdf(r, n, e) ** 2:.3g}")
for r, a in [(800, .045), (800, .06), (900, .05)]:
    q = e + (1 - 2 * e) * a
    print(f"  Fig.2c below the 1e-14 clip: r={r}, a={a}: log10 P_raw = {(2 * stats.binom.logcdf(r, n, q) + 2 * stats.binom.logcdf(r, n, e)) / math.log(10):.1f}")
print(f"  pmf peak = {stats.binom.pmf(round(n * qX), n, qX):.4f}; 0/1e5 95% UB = {1 - .05 ** 1e-5:.3g}")
