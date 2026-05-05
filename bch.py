from dataclasses import dataclass

from .galois import GaloisField
from .polynomial import Polynomial, lcm_poly_gf2


@dataclass
class DecodeResult:
    success: bool
    decoded_codeword: list[int]
    decoded_message: list[int]
    error_positions: list[int]
    syndromes: list[int]
    error_locator: list[int]

    @property
    def num_errors_corrected(self) -> int:
        return len(self.error_positions)


class BCHCode:
    """Binary BCH code with parameters (n, k, t) over GF(2^m)."""

    def __init__(self, m: int, t: int):
        if t < 1:
            raise ValueError(f"t must be >= 1, got {t}")
        self.gf = GaloisField(m)
        self.m = m
        self.n = 2 ** m - 1
        self.t = t

        if 2 * t >= self.n:
            raise ValueError(f"t={t} too large: need 2t < n = {self.n}")

        # g(x) = lcm of minimal polynomials of alpha, alpha^2, ..., alpha^(2t)
        min_polys = []
        seen_cosets = set()
        for i in range(1, 2 * t + 1):
            coset = frozenset(self._cyclotomic_coset(i))
            if coset in seen_cosets:
                continue
            seen_cosets.add(coset)
            min_polys.append(self.gf.minimal_polynomial(self.gf.alpha(i)))

        g_coeffs = lcm_poly_gf2(min_polys)
        self.generator = Polynomial(g_coeffs)
        self.k = self.n - self.generator.degree

        if self.k <= 0:
            raise ValueError(f"resulting k={self.k} is non-positive")

    def __repr__(self) -> str:
        return f"BCH(n={self.n}, k={self.k}, t={self.t}, m={self.m})"

    def _cyclotomic_coset(self, i: int) -> list[int]:
        coset = []
        x = i % self.n if self.n > 0 else 0
        while x not in coset:
            coset.append(x)
            x = (x * 2) % self.n
        return coset

    @property
    def code_rate(self) -> float:
        return self.k / self.n

    def parity_check_matrix(self) -> list[list[int]]:
        """Symbolic parity-check matrix H (2t x n) over GF(2^m).

        H[i][j] = alpha^((i+1) * j). A vector r is a codeword iff H * r^T = 0.
        """
        H = []
        for i in range(1, 2 * self.t + 1):
            row = [self.gf.alpha((i * j) % self.gf.order) for j in range(self.n)]
            H.append(row)
        return H

    def generator_matrix(self) -> list[list[int]]:
        """Systematic generator matrix G (k x n) over GF(2)."""
        G = []
        for i in range(self.k):
            unit = [0] * self.k
            unit[i] = 1
            G.append(self.encode(unit))
        return G

    def encode(self, message: list[int]) -> list[int]:
        if len(message) != self.k:
            raise ValueError(f"message length must be {self.k}, got {len(message)}")
        for b in message:
            if b not in (0, 1):
                raise ValueError(f"message bits must be 0 or 1, got {b}")

        # c(x) = x^(n-k) * m(x) + [x^(n-k) * m(x)] mod g(x)
        n_k = self.n - self.k
        msg_low = list(reversed(message))
        shifted = [0] * n_k + msg_low
        shifted_poly = Polynomial(shifted)
        _, remainder = shifted_poly.divmod(self.generator)

        codeword_low = [0] * self.n
        for i, c in enumerate(remainder.coeffs):
            codeword_low[i] = c
        for i, c in enumerate(msg_low):
            codeword_low[n_k + i] = c

        return list(reversed(codeword_low))

    def decode(self, received: list[int]) -> DecodeResult:
        if len(received) != self.n:
            raise ValueError(f"received length must be {self.n}, got {len(received)}")
        for b in received:
            if b not in (0, 1):
                raise ValueError(f"received bits must be 0 or 1, got {b}")

        r_coeffs = [received[self.n - 1 - i] for i in range(self.n)]

        syndromes = []
        for i in range(1, 2 * self.t + 1):
            s = 0
            ai = self.gf.alpha(i)
            x_pow = 1
            for c in r_coeffs:
                if c:
                    s ^= x_pow
                x_pow = self.gf.mul(x_pow, ai)
            syndromes.append(s)

        if all(s == 0 for s in syndromes):
            return DecodeResult(
                success=True,
                decoded_codeword=list(received),
                decoded_message=list(received[:self.k]),
                error_positions=[],
                syndromes=syndromes,
                error_locator=[1],
            )

        locator = self._berlekamp_massey(syndromes)

        error_positions_low = []
        for j in range(self.n):
            x = self.gf.alpha((-j) % self.gf.order)
            val = 0
            x_pow = 1
            for c in locator:
                val ^= self.gf.mul(c, x_pow)
                x_pow = self.gf.mul(x_pow, x)
            if val == 0:
                error_positions_low.append(j)

        error_positions_msb = [self.n - 1 - p for p in error_positions_low]

        if len(error_positions_low) == 0 or len(error_positions_low) > self.t:
            return DecodeResult(
                success=False,
                decoded_codeword=list(received),
                decoded_message=list(received[:self.k]),
                error_positions=error_positions_msb,
                syndromes=syndromes,
                error_locator=locator,
            )

        corrected = list(received)
        for p in error_positions_msb:
            corrected[p] ^= 1

        # verify by recomputing syndromes
        c_coeffs = [corrected[self.n - 1 - i] for i in range(self.n)]
        for i in range(1, 2 * self.t + 1):
            s = 0
            ai = self.gf.alpha(i)
            x_pow = 1
            for c in c_coeffs:
                if c:
                    s ^= x_pow
                x_pow = self.gf.mul(x_pow, ai)
            if s != 0:
                return DecodeResult(
                    success=False,
                    decoded_codeword=list(received),
                    decoded_message=list(received[:self.k]),
                    error_positions=error_positions_msb,
                    syndromes=syndromes,
                    error_locator=locator,
                )

        return DecodeResult(
            success=True,
            decoded_codeword=corrected,
            decoded_message=corrected[:self.k],
            error_positions=error_positions_msb,
            syndromes=syndromes,
            error_locator=locator,
        )

    def _berlekamp_massey(self, syndromes: list[int]) -> list[int]:
        """Find minimal-degree error-locator polynomial Lambda(x)."""
        gf = self.gf
        L = 0
        Lambda = [1]
        B = [1]
        b = 1
        m_shift = 1

        for n in range(len(syndromes)):
            d = syndromes[n]
            for i in range(1, L + 1):
                if i < len(Lambda):
                    d ^= gf.mul(Lambda[i], syndromes[n - i])

            if d == 0:
                m_shift += 1
            elif 2 * L <= n:
                T = list(Lambda)
                coef = gf.div(d, b)
                shifted_B = [0] * m_shift + B
                while len(Lambda) < len(shifted_B):
                    Lambda.append(0)
                for i, bi in enumerate(shifted_B):
                    if i < len(Lambda):
                        Lambda[i] ^= gf.mul(coef, bi)
                L = n + 1 - L
                B = T
                b = d
                m_shift = 1
            else:
                coef = gf.div(d, b)
                shifted_B = [0] * m_shift + B
                while len(Lambda) < len(shifted_B):
                    Lambda.append(0)
                for i, bi in enumerate(shifted_B):
                    if i < len(Lambda):
                        Lambda[i] ^= gf.mul(coef, bi)
                m_shift += 1

        while len(Lambda) > 1 and Lambda[-1] == 0:
            Lambda.pop()

        return Lambda
