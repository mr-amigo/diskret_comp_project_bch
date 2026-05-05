"""Polynomials over GF(2) and GF(2^m)."""


class Polynomial:
    """Polynomial with coefficients (low-degree first). Optional GaloisField."""

    def __init__(self, coeffs, field=None):
        coeffs = list(coeffs)
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        if not coeffs:
            coeffs = [0]
        self.coeffs = coeffs
        self.field = field

    @property
    def degree(self):
        if self.is_zero():
            return -1
        return len(self.coeffs) - 1

    def is_zero(self):
        return len(self.coeffs) == 1 and self.coeffs[0] == 0

    def __getitem__(self, i):
        if 0 <= i < len(self.coeffs):
            return self.coeffs[i]
        return 0

    def __add__(self, other):
        if not isinstance(other, Polynomial):
            return NotImplemented
        if self.field != other.field:
            raise ValueError("polynomials must have the same field")
        n = max(len(self.coeffs), len(other.coeffs))
        result = [0] * n
        for i in range(n):
            result[i] = self[i] ^ other[i]
        return Polynomial(result, self.field)

    def __sub__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, Polynomial):
            return NotImplemented
        if self.field != other.field:
            raise ValueError("polynomials must have the same field")
        if self.is_zero() or other.is_zero():
            return Polynomial([0], self.field)

        n = len(self.coeffs) + len(other.coeffs) - 1
        result = [0] * n
        if self.field is None:
            for i, a in enumerate(self.coeffs):
                if a:
                    for j, b in enumerate(other.coeffs):
                        if b:
                            result[i + j] ^= 1
        else:
            for i, a in enumerate(self.coeffs):
                if a == 0:
                    continue
                for j, b in enumerate(other.coeffs):
                    if b == 0:
                        continue
                    result[i + j] ^= self.field.mul(a, b)
        return Polynomial(result, self.field)

    def divmod(self, divisor):
        if divisor.is_zero():
            raise ZeroDivisionError("polynomial division by zero")
        if self.degree < divisor.degree:
            return Polynomial([0], self.field), Polynomial(self.coeffs, self.field)

        remainder = list(self.coeffs)
        quotient = [0] * (self.degree - divisor.degree + 1)
        d_lead = divisor.coeffs[-1]
        d_deg = divisor.degree

        for i in range(len(quotient) - 1, -1, -1):
            if i + d_deg >= len(remainder):
                continue
            r_lead = remainder[i + d_deg]
            if r_lead == 0:
                continue
            if self.field is None:
                coef = r_lead
            else:
                coef = self.field.div(r_lead, d_lead)
            quotient[i] = coef
            for j in range(d_deg + 1):
                if self.field is None:
                    remainder[i + j] ^= divisor.coeffs[j] & coef
                else:
                    remainder[i + j] ^= self.field.mul(divisor.coeffs[j], coef)

        return Polynomial(quotient, self.field), Polynomial(remainder, self.field)

    def __mod__(self, other):
        return self.divmod(other)[1]

    def __floordiv__(self, other):
        return self.divmod(other)[0]

    def eval(self, x):
        result = 0
        x_pow = 1
        for c in self.coeffs:
            if c:
                if self.field is None:
                    result ^= c * x_pow if x_pow == 1 or x_pow == 0 else (c * x_pow) & 1
                else:
                    result ^= self.field.mul(c, x_pow)
            if self.field is None:
                x_pow = x_pow * x if x_pow else 0
            else:
                x_pow = self.field.mul(x_pow, x)
        return result

    def __repr__(self):
        return f"Polynomial({self.coeffs})"


def lcm_poly_gf2(polys):
    """Least common multiple of GF(2) polynomials, returned as coefficient list."""
    result = Polynomial([1])
    for p in polys:
        poly = Polynomial(list(p))
        gcd_poly = _gcd_gf2(result, poly)
        result = (result * poly) // gcd_poly
    return result.coeffs


def _gcd_gf2(a, b):
    while not b.is_zero():
        a, b = b, a % b
    return a
