"""Edge case tests — boundary conditions and regression checks."""

import pytest
import random
from core.galois import GaloisField
from core.polynomial import Polynomial, lcm_poly_gf2
from core.bch import BCHCode
from core.channel import (
    BinarySymmetricChannel,
    BurstErrorChannel,
    AWGNChannel,
)


# =============== GaloisField edge cases ===============

class TestGFEdgeCases:
    def test_m_too_small_raises(self):
        """m=1 (GF(2)) is trivial and not supported."""
        with pytest.raises(ValueError, match="m must be in"):
            GaloisField(1)

    def test_m_too_large_raises(self):
        with pytest.raises(ValueError):
            GaloisField(17)

    def test_non_primitive_polynomial_detected(self):
        """x^2 + 1 = (x+1)^2 over GF(2) — reducible, not primitive."""
        with pytest.raises(ValueError, match="not primitive"):
            GaloisField(2, 0b101)

    def test_wrong_degree_polynomial_raises(self):
        with pytest.raises(ValueError, match="degree exactly"):
            GaloisField(4, 0b101)  # only degree 2

    def test_pow_zero_zero(self):
        """0^0 = 1 by convention (matches Python)."""
        gf = GaloisField(4)
        assert gf.pow(0, 0) == 1

    def test_pow_zero_positive(self):
        gf = GaloisField(4)
        assert gf.pow(0, 5) == 0

    def test_pow_negative(self):
        gf = GaloisField(4)
        assert gf.pow(5, -1) == gf.inv(5)

    def test_alpha_huge_wraps(self):
        gf = GaloisField(4)
        assert gf.alpha(15 * 100) == 1

    def test_alpha_negative(self):
        gf = GaloisField(4)
        assert gf.alpha(-1) == gf.inv(gf.alpha(1))

    def test_log_zero_raises(self):
        gf = GaloisField(4)
        with pytest.raises(ValueError):
            gf.log(0)

    def test_log_one(self):
        gf = GaloisField(4)
        assert gf.log(1) == 0

    def test_div_by_zero_raises(self):
        gf = GaloisField(4)
        with pytest.raises(ZeroDivisionError):
            gf.div(5, 0)

    def test_div_zero_by_anything(self):
        gf = GaloisField(4)
        for a in range(1, 16):
            assert gf.div(0, a) == 0

    def test_element_repr_special(self):
        gf = GaloisField(4)
        assert gf.element_repr(0) == "0"
        assert gf.element_repr(1) == "1"
        assert gf.element_repr(gf.alpha(1)) == "α"
        assert "α^" in gf.element_repr(gf.alpha(7))

    def test_field_equality(self):
        gf1 = GaloisField(4)
        gf2 = GaloisField(4)
        gf3 = GaloisField(5)
        assert gf1 == gf2
        assert gf1 != gf3
        assert hash(gf1) == hash(gf2)


# =============== Polynomial edge cases ===============

class TestPolynomialEdgeCases:
    def test_empty_list_is_zero(self):
        p = Polynomial([])
        assert p.is_zero()
        assert p.degree == -1

    def test_zero_polynomial_degree(self):
        assert Polynomial([0]).degree == -1
        assert Polynomial([0, 0, 0]).degree == -1

    def test_trailing_zeros_stripped(self):
        assert Polynomial([1, 0, 0]).coeffs == [1]
        assert Polynomial([1, 1, 0]).coeffs == [1, 1]

    def test_add_zero_is_identity(self):
        p = Polynomial([1, 1, 1])
        z = Polynomial([0])
        assert (p + z).coeffs == p.coeffs

    def test_mul_by_zero(self):
        p = Polynomial([1, 1, 1])
        z = Polynomial([0])
        assert (p * z).is_zero()

    def test_eval_zero_at_anywhere(self):
        z = Polynomial([0])
        for x in [0, 1, 5, 100]:
            assert z.eval(x) == 0

    def test_derivative_constant(self):
        # In char 2, derivative of x^k for even k is 0
        assert Polynomial([5]).derivative().is_zero()
        assert Polynomial([0, 0, 1]).derivative().is_zero()  # x^2 -> 0
        assert Polynomial([0, 0, 0, 0, 1]).derivative().is_zero()  # x^4 -> 0

    def test_derivative_odd_powers(self):
        # x -> 1
        assert Polynomial([0, 1]).derivative().coeffs == [1]
        # x^3 + x -> x^2 + 1 (in char 2: 3x^2 = x^2, 1 = 1)
        assert Polynomial([0, 1, 0, 1]).derivative().coeffs == [1, 0, 1]

    def test_divmod_by_self(self):
        p = Polynomial([1, 0, 1])
        q, r = p.divmod(p)
        assert r.is_zero()
        assert q.coeffs == [1]

    def test_divmod_by_one(self):
        p = Polynomial([1, 1, 1, 0, 1])
        q, r = p.divmod(Polynomial([1]))
        assert r.is_zero()
        assert q.coeffs == p.coeffs

    def test_divmod_smaller(self):
        small = Polynomial([1])
        big = Polynomial([0, 1])  # x
        q, r = small.divmod(big)
        assert q.is_zero()
        assert r.coeffs == small.coeffs

    def test_divmod_by_zero_raises(self):
        p = Polynomial([1, 1])
        with pytest.raises(ZeroDivisionError):
            p.divmod(Polynomial([0]))

    def test_field_mismatch_raises(self):
        gf = GaloisField(4)
        p1 = Polynomial([1], gf)
        p2 = Polynomial([1])
        with pytest.raises(ValueError):
            p1 + p2

    def test_lcm_single(self):
        assert lcm_poly_gf2([[1, 1]]) == [1, 1]

    def test_lcm_duplicates(self):
        assert lcm_poly_gf2([[1, 1], [1, 1]]) == [1, 1]


# =============== BCH edge cases ===============

class TestBCHEdgeCases:
    def test_t_zero_raises(self):
        with pytest.raises(ValueError, match="t must be"):
            BCHCode(m=4, t=0)

    def test_t_too_large_raises(self):
        with pytest.raises(ValueError):
            BCHCode(m=3, t=4)  # 2*4 >= 7

    def test_large_m(self):
        """m up to 12 should build successfully."""
        BCHCode(m=10, t=5)
        BCHCode(m=12, t=10)

    def test_max_t_for_m(self):
        """The largest t such that 2t < n should still work."""
        # For m=4 (n=15), max t is 7 (2*7=14 < 15)
        # but k may be 0 or even negative, which we should reject
        with pytest.raises(ValueError):
            BCHCode(m=4, t=8)  # too big

    def test_burst_errors_within_capacity(self):
        bch = BCHCode(m=5, t=3)
        rng = random.Random(0)
        # 3 consecutive errors (a small burst)
        for trial in range(10):
            msg = [rng.randint(0, 1) for _ in range(bch.k)]
            cw = bch.encode(msg)
            start = rng.randint(0, bch.n - 3)
            received = list(cw)
            for i in range(3):
                received[start + i] ^= 1
            res = bch.decode(received)
            assert res.success
            assert res.decoded_message == msg

    def test_errors_only_in_parity(self):
        bch = BCHCode(m=5, t=3)
        rng = random.Random(1)
        msg = [rng.randint(0, 1) for _ in range(bch.k)]
        cw = bch.encode(msg)
        # parity bits are at positions [k..n-1]
        positions = rng.sample(range(bch.k, bch.n), bch.t)
        received = list(cw)
        for p in positions:
            received[p] ^= 1
        res = bch.decode(received)
        assert res.success
        assert res.decoded_message == msg

    def test_errors_only_in_message(self):
        bch = BCHCode(m=5, t=3)
        rng = random.Random(2)
        msg = [rng.randint(0, 1) for _ in range(bch.k)]
        cw = bch.encode(msg)
        positions = rng.sample(range(bch.k), bch.t)
        received = list(cw)
        for p in positions:
            received[p] ^= 1
        res = bch.decode(received)
        assert res.success
        assert res.decoded_message == msg

    def test_all_bits_errored_no_crash(self):
        """Decoder must handle n errors gracefully — not crash."""
        bch = BCHCode(m=5, t=3)
        msg = [0] * bch.k
        cw = bch.encode(msg)
        received = [1 - b for b in cw]
        res = bch.decode(received)
        # Either success=False or mis-correction; just no crash.
        assert isinstance(res.success, bool)

    def test_decoded_codeword_is_valid(self):
        """When decoder succeeds, decoded codeword must satisfy c(α^i) = 0."""
        bch = BCHCode(m=5, t=3)
        rng = random.Random(0)
        for _ in range(20):
            msg = [rng.randint(0, 1) for _ in range(bch.k)]
            cw = bch.encode(msg)
            positions = rng.sample(range(bch.n), 2)
            received = list(cw)
            for p in positions:
                received[p] ^= 1
            res = bch.decode(received)
            if res.success:
                # check codeword property
                cw_low = list(reversed(res.decoded_codeword))
                for i in range(1, 2 * bch.t + 1):
                    s = 0
                    ai = bch.gf.alpha(i)
                    xp = 1
                    for c in cw_low:
                        if c:
                            s ^= xp
                        xp = bch.gf.mul(xp, ai)
                    assert s == 0, f"S_{i} != 0 after decode"

    def test_error_position_msb_first_helper(self):
        bch = BCHCode(m=4, t=2)
        msg = [1, 0, 1, 0, 1, 0, 1]
        cw = bch.encode(msg)
        received = list(cw)
        received[3] ^= 1
        received[10] ^= 1
        res = bch.decode(received)
        assert res.success
        msb_pos = res.error_positions_msb_first(bch.n)
        assert sorted(msb_pos) == [3, 10]

    def test_encode_idempotent(self):
        """Encoding the same message twice yields the same codeword."""
        bch = BCHCode(m=5, t=3)
        msg = [1, 0, 1, 0, 1, 0, 1, 0] * 2
        cw1 = bch.encode(msg)
        cw2 = bch.encode(msg)
        assert cw1 == cw2

    def test_encode_decode_idempotent_no_errors(self):
        bch = BCHCode(m=5, t=3)
        rng = random.Random(0)
        for _ in range(10):
            msg = [rng.randint(0, 1) for _ in range(bch.k)]
            cw = bch.encode(msg)
            res = bch.decode(cw)
            cw2 = bch.encode(res.decoded_message)
            assert cw == cw2


# =============== Channel edge cases ===============

class TestChannelEdgeCases:
    def test_bsc_empty_input(self):
        ch = BinarySymmetricChannel(p=0.5)
        stats = ch.transmit([])
        assert stats.received == []
        assert stats.num_errors == 0

    def test_bsc_reproducibility(self):
        bits = [random.Random(0).randint(0, 1) for _ in range(100)]
        s1 = BinarySymmetricChannel(0.3, seed=42).transmit(bits)
        s2 = BinarySymmetricChannel(0.3, seed=42).transmit(bits)
        assert s1.received == s2.received

    def test_bsc_exact_zero(self):
        ch = BinarySymmetricChannel(p=0.5, seed=0)
        stats = ch.transmit_with_exact_errors([0] * 10, 0)
        assert stats.num_errors == 0
        assert stats.received == [0] * 10

    def test_bsc_exact_all(self):
        ch = BinarySymmetricChannel(p=0.5, seed=0)
        stats = ch.transmit_with_exact_errors([0] * 10, 10)
        assert stats.received == [1] * 10

    def test_burst_length_zero_raises(self):
        with pytest.raises(ValueError):
            BurstErrorChannel(burst_length=0, burst_rate=0.1)

    def test_burst_invalid_rate_raises(self):
        with pytest.raises(ValueError):
            BurstErrorChannel(burst_length=5, burst_rate=1.5)

    def test_burst_clipped_at_end(self):
        ch = BurstErrorChannel(burst_length=10, burst_rate=0)
        stats = ch.transmit_one_burst([0] * 15, position=12)
        assert stats.error_positions == [12, 13, 14]

    def test_awgn_q_function_match(self):
        """Equivalent BER should match the Q-function exactly."""
        import math
        for snr in [0, 3, 5, 8, 10]:
            ch = AWGNChannel(snr_db=snr)
            expected = 0.5 * math.erfc(1.0 / (ch.sigma * math.sqrt(2)))
            assert abs(ch.equivalent_ber - expected) < 1e-10

    def test_awgn_zero_db(self):
        """At 0 dB, BER ≈ Q(sqrt(2)) ≈ 0.0786."""
        ch = AWGNChannel(snr_db=0.0)
        assert abs(ch.equivalent_ber - 0.0786) < 0.001

    def test_awgn_reproducibility(self):
        bits = [0, 1] * 50
        s1 = AWGNChannel(5, seed=42).transmit(bits)
        s2 = AWGNChannel(5, seed=42).transmit(bits)
        assert s1.received == s2.received
