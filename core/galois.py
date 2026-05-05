"""Galois field GF(2^m) with log/antilog tables for fast multiplication."""

PRIMITIVE_POLYS = {
    2: 0b111,
    3: 0b1011,
    4: 0b10011,
    5: 0b100101,
    6: 0b1000011,
    7: 0b10001001,
    8: 0b100011101,
    9: 0b1000010001,
    10: 0b10000001001,
}


class GaloisField:
    def __init__(self, m: int):
        if not (2 <= m <= 10):
            raise ValueError(f"m must be in [2, 10], got {m}")
        self.m = m
        self.size = 1 << m
        self.order = self.size - 1
        self.prim_poly = PRIMITIVE_POLYS[m]
        self._build_tables()

    def _build_tables(self):
        order = self.order
        antilog = [0] * (2 * order)
        log = [-1] * self.size
        x = 1
        high_bit = 1 << self.m
        for i in range(order):
            antilog[i] = x
            log[x] = i
            x <<= 1
            if x & high_bit:
                x ^= self.prim_poly
        for i in range(order):
            antilog[order + i] = antilog[i]
        self._log = log
        self._antilog = antilog

    def add(self, a, b):
        return a ^ b

    def mul(self, a, b):
        if a == 0 or b == 0:
            return 0
        return self._antilog[self._log[a] + self._log[b]]

    def div(self, a, b):
        if b == 0:
            raise ZeroDivisionError("division by zero in GF(2^m)")
        if a == 0:
            return 0
        return self._antilog[(self._log[a] - self._log[b]) % self.order]

    def inv(self, a):
        if a == 0:
            raise ZeroDivisionError("0 has no inverse")
        return self._antilog[self.order - self._log[a]]

    def pow(self, a, k):
        if a == 0:
            return 0 if k > 0 else 1
        return self._antilog[(self._log[a] * k) % self.order]

    def alpha(self, k=1):
        return self._antilog[k % self.order]

    def log(self, a):
        if a == 0:
            raise ValueError("log(0) is undefined")
        return self._log[a]

    def element_repr(self, a):
        if a == 0:
            return "0"
        k = self._log[a]
        if k == 0:
            return "1"
        if k == 1:
            return "a"
        return f"a^{k}"

    def minimal_polynomial(self, a):
        """Minimal polynomial of `a` over GF(2). Returns coeffs (low-degree first)."""
        if a == 0:
            return [0, 1]

        conjugates = [a]
        cur = self.mul(a, a)
        while cur != a:
            conjugates.append(cur)
            cur = self.mul(cur, cur)

        poly = [1]
        for c in conjugates:
            new_poly = [0] * (len(poly) + 1)
            for i, p in enumerate(poly):
                new_poly[i] ^= self.mul(p, c)
                new_poly[i + 1] ^= p
            poly = new_poly

        return poly

    def __repr__(self):
        return f"GF(2^{self.m})"
