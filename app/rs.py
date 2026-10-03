"""Reed-Solomon error correction over GF(2^8).

Pure-python implementation (primitive polynomial 0x11D, first consecutive
root 0) used to protect the watermark payload against JPEG / scaling /
cropping damage.  The decoder runs Berlekamp-Massey + Chien search +
Forney and raises ``RSDecodeError`` when the codeword is damaged beyond
correction capability.

Conventions
-----------
* Codewords are lists of bytes, first byte = highest-degree coefficient
  (polynomial evaluated in descending order).
* Syndrome j is ``r(alpha**j)`` for j in 0..nsym-1.
* Locator/evaluator polynomials are kept in *ascending* coefficient order.
"""

PRIMITIVE = 0x11D
FIELD_SIZE = 255

# ---------------------------------------------------------------------------
# GF(2^8) arithmetic
# ---------------------------------------------------------------------------

GF_EXP = [0] * 512
GF_LOG = [0] * 256

_x = 1
for _i in range(FIELD_SIZE):
    GF_EXP[_i] = _x
    GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= PRIMITIVE
for _i in range(FIELD_SIZE, 512):
    GF_EXP[_i] = GF_EXP[_i - FIELD_SIZE]


def gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return GF_EXP[GF_LOG[a] + GF_LOG[b]]


def gf_div(a, b):
    if b == 0:
        raise ZeroDivisionError("GF division by zero")
    if a == 0:
        return 0
    return GF_EXP[(GF_LOG[a] + FIELD_SIZE - GF_LOG[b]) % FIELD_SIZE]


def gf_pow(a, n):
    if n == 0:
        return 1
    if a == 0:
        return 0
    return GF_EXP[(GF_LOG[a] * n) % FIELD_SIZE]


def gf_poly_mul(p, q):
    """Multiply two polynomials (descending coefficient order)."""
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                if b:
                    out[i + j] ^= gf_mul(a, b)
    return out


def gf_poly_eval(poly, x):
    """Evaluate a polynomial (descending coefficient order) at ``x``."""
    y = poly[0]
    for coef in poly[1:]:
        y = gf_mul(y, x) ^ coef
    return y


def _gf_poly_eval_asc(poly, x):
    """Evaluate a polynomial (ascending coefficient order) at ``x``."""
    y = 0
    for coef in reversed(poly):
        y = gf_mul(y, x) ^ coef
    return y


# ---------------------------------------------------------------------------
# Reed-Solomon codec
# ---------------------------------------------------------------------------

class RSDecodeError(Exception):
    """Raised when a codeword cannot be corrected."""


def rs_generator_poly(nsym):
    """Generator polynomial (descending order) for ``nsym`` parity symbols."""
    g = [1]
    for i in range(nsym):
        g = gf_poly_mul(g, [1, GF_EXP[i]])
    return g


def rs_encode(msg, nsym):
    """Return ``msg`` (list of ints) followed by ``nsym`` parity bytes."""
    if nsym < 1:
        raise ValueError("nsym must be positive")
    gen = rs_generator_poly(nsym)
    out = list(msg) + [0] * nsym
    for i in range(len(msg)):
        coef = out[i]
        if coef:
            for j in range(1, len(gen)):
                out[i + j] ^= gf_mul(gen[j], coef)
    return list(msg) + out[len(msg):]


def _rs_syndromes(msg, nsym):
    """S_j = r(alpha^j), j = 0..nsym-1 (ascending list)."""
    return [gf_poly_eval(msg, GF_EXP[j]) for j in range(nsym)]


def _rs_berlekamp_massey(synd, nsym):
    """Standard BM; returns error locator in ascending coefficient order."""
    lam = [1]  # connection polynomial C(x)
    b = [1]    # last connection polynomial B(x)
    L = 0
    m = 1
    bb = 1     # last nonzero discrepancy
    for n in range(nsym):
        d = synd[n]
        for i in range(1, L + 1):
            if i < len(lam):
                d ^= gf_mul(lam[i], synd[n - i])
        if d == 0:
            m += 1
        elif 2 * L <= n:
            t = list(lam)
            coef = gf_div(d, bb)
            # lam = lam - coef * x^m * B
            if len(lam) < len(b) + m:
                lam = lam + [0] * (len(b) + m - len(lam))
            for i, v in enumerate(b):
                lam[i + m] ^= gf_mul(coef, v)
            L = n + 1 - L
            b = t
            bb = d
            m = 1
        else:
            coef = gf_div(d, bb)
            if len(lam) < len(b) + m:
                lam = lam + [0] * (len(b) + m - len(lam))
            for i, v in enumerate(b):
                lam[i + m] ^= gf_mul(coef, v)
            m += 1
    if L * 2 > nsym:
        raise RSDecodeError("too many errors to correct")
    return lam[:L + 1] if L > 0 else [1]


def _rs_chien_search(lam, nmess):
    """Return positions p (0 = first byte) where errors occurred."""
    err_pos = []
    for p in range(nmess):
        # root of Lambda(x) at x = alpha^{-(nmess-1-p)}
        x = GF_EXP[(p + 1 + (FIELD_SIZE - nmess)) % FIELD_SIZE]
        if _gf_poly_eval_asc(lam, x) == 0:
            err_pos.append(p)
    if len(err_pos) != len(lam) - 1:
        raise RSDecodeError("could not locate errors")
    return err_pos


def _rs_forney(msg, synd, lam, err_pos, nmess):
    """Compute error magnitudes and apply corrections (in place copy)."""
    # Omega(x) = (Lambda(x) * S(x)) mod x^nsym  (ascending order)
    nsym = len(synd)
    omega = [0] * nsym
    for i, s in enumerate(synd):
        if s:
            for j, c in enumerate(lam):
                if c and i + j < nsym:
                    omega[i + j] ^= gf_mul(s, c)
    # Lambda'(x): formal derivative (ascending); in GF(2) only odd powers
    # survive, and coefficient lam[i] lands at position i-1.
    lam_prime = [(lam[i] if i % 2 == 1 else 0) for i in range(1, len(lam))]
    out = list(msg)
    for p in err_pos:
        x_inv = GF_EXP[(p + 1 + (FIELD_SIZE - nmess)) % FIELD_SIZE]  # X^-1
        x_loc = GF_EXP[(nmess - 1 - p) % FIELD_SIZE]                 # X
        num = _gf_poly_eval_asc(omega, x_inv)
        den = _gf_poly_eval_asc(lam_prime, x_inv)
        if den == 0:
            raise RSDecodeError("singular Forney denominator")
        magnitude = gf_mul(x_loc, gf_div(num, den))
        out[p] ^= magnitude
    return out


def rs_decode(codeword, nsym):
    """Correct and return ``(message, num_errors)``.

    Raises ``RSDecodeError`` if correction fails.
    """
    if len(codeword) <= nsym:
        raise RSDecodeError("codeword shorter than parity")
    msg = list(codeword)
    synd = _rs_syndromes(msg, nsym)
    if max(synd) == 0:
        return msg[:-nsym], 0
    lam = _rs_berlekamp_massey(synd, nsym)
    err_pos = _rs_chien_search(lam, len(msg))
    msg = _rs_forney(msg, synd, lam, err_pos, len(msg))
    if max(_rs_syndromes(msg, nsym)) != 0:
        raise RSDecodeError("correction failed (residual syndromes)")
    return msg[:-nsym], len(err_pos)
