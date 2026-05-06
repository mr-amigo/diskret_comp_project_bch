"""Tests for channel models: BSC, Burst, AWGN."""

import math
import random
import pytest
from core.channel import (
    BinarySymmetricChannel,
    BurstErrorChannel,
    AWGNChannel,
    ChannelStats,
)


class TestBSC:
    def test_p_zero_no_errors(self):
        ch = BinarySymmetricChannel(p=0.0, seed=0)
        bits = [0, 1, 0, 1, 1, 0]
        stats = ch.transmit(bits)
        assert stats.received == bits
        assert stats.num_errors == 0

    def test_p_one_all_flipped(self):
        ch = BinarySymmetricChannel(p=1.0, seed=0)
        stats = ch.transmit([0, 1, 0, 1, 1])
        assert stats.received == [1, 0, 1, 0, 0]
        assert stats.num_errors == 5

    def test_empty_input(self):
        ch = BinarySymmetricChannel(p=0.5, seed=0)
        stats = ch.transmit([])
        assert stats.received == []
        assert stats.num_errors == 0
        assert stats.empirical_ber == 0  # by convention (no division by zero)

    def test_reproducibility(self):
        bits = [0, 1] * 100
        s1 = BinarySymmetricChannel(0.3, seed=42).transmit(bits)
        s2 = BinarySymmetricChannel(0.3, seed=42).transmit(bits)
        assert s1.received == s2.received

    def test_empirical_ber_close_to_p(self):
        ch = BinarySymmetricChannel(p=0.1, seed=0)
        stats = ch.transmit([0] * 10000)
        assert abs(stats.empirical_ber - 0.1) < 0.02

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            BinarySymmetricChannel(p=-0.1)
        with pytest.raises(ValueError):
            BinarySymmetricChannel(p=1.5)

    def test_exact_errors_zero(self):
        ch = BinarySymmetricChannel(p=0.5, seed=0)
        stats = ch.transmit_with_exact_errors([0] * 10, 0)
        assert stats.num_errors == 0

    def test_exact_errors_all(self):
        ch = BinarySymmetricChannel(p=0.5, seed=0)
        stats = ch.transmit_with_exact_errors([0] * 10, 10)
        assert stats.received == [1] * 10

    def test_exact_errors_too_many(self):
        ch = BinarySymmetricChannel(p=0.5, seed=0)
        with pytest.raises(ValueError):
            ch.transmit_with_exact_errors([0] * 5, 6)


class TestBurst:
    def test_invalid_burst_length(self):
        with pytest.raises(ValueError):
            BurstErrorChannel(burst_length=0, burst_rate=0.1)

    def test_invalid_burst_rate(self):
        with pytest.raises(ValueError):
            BurstErrorChannel(burst_length=5, burst_rate=1.5)

    def test_one_burst(self):
        ch = BurstErrorChannel(burst_length=5, burst_rate=0, seed=0)
        bits = [0] * 50
        stats = ch.transmit_one_burst(bits, position=10)
        assert stats.error_positions == [10, 11, 12, 13, 14]

    def test_burst_clipped_at_end(self):
        ch = BurstErrorChannel(burst_length=10, burst_rate=0, seed=0)
        bits = [0] * 20
        stats = ch.transmit_one_burst(bits, position=15)
        assert stats.error_positions == [15, 16, 17, 18, 19]

    def test_zero_rate_no_bursts(self):
        ch = BurstErrorChannel(burst_length=5, burst_rate=0.0, seed=0)
        bits = [0] * 100
        stats = ch.transmit(bits)
        assert stats.num_errors == 0


class TestAWGN:
    def test_high_snr_low_error(self):
        ch = AWGNChannel(snr_db=20.0)
        assert ch.equivalent_ber < 1e-5

    def test_low_snr_high_error(self):
        ch = AWGNChannel(snr_db=-10.0)
        assert ch.equivalent_ber > 0.2

    def test_zero_db_q_function_value(self):
        # At 0 dB, BER = Q(sqrt(2)) ≈ 0.0786
        ch = AWGNChannel(snr_db=0.0)
        assert abs(ch.equivalent_ber - 0.0786) < 0.001

    def test_equivalent_ber_matches_q_function(self):
        for snr in [0, 3, 5, 8, 10]:
            ch = AWGNChannel(snr_db=snr)
            expected = 0.5 * math.erfc(1.0 / (ch.sigma * math.sqrt(2)))
            assert abs(ch.equivalent_ber - expected) < 1e-10

    def test_reproducibility(self):
        bits = [random.Random(0).randint(0, 1) for _ in range(100)]
        s1 = AWGNChannel(snr_db=5.0, seed=42).transmit(bits)
        s2 = AWGNChannel(snr_db=5.0, seed=42).transmit(bits)
        assert s1.received == s2.received

    def test_transmission_at_moderate_snr(self):
        ch = AWGNChannel(snr_db=3.0, seed=42)
        bits = [random.Random(0).randint(0, 1) for _ in range(1000)]
        stats = ch.transmit(bits)
        assert len(stats.received) == 1000
        # at 3 dB, BER ~ 0.023, expect 5..200 errors out of 1000
        assert 0 < stats.num_errors < 200


class TestChannelStats:
    def test_stats_fields(self):
        s = ChannelStats(
            transmitted=[0, 1, 0],
            received=[0, 0, 1],
            error_positions=[1, 2],
        )
        assert s.num_errors == 2
        assert s.empirical_ber == pytest.approx(2 / 3)


class TestChannelExtremes:
    """Extreme parameter values — should not crash, should give sensible answers."""

    def test_awgn_at_50db_zero_errors(self):
        """At extreme SNR, AWGN should pass bits through perfectly."""
        ch = AWGNChannel(snr_db=50.0, seed=0)
        bits = [0, 1] * 500
        stats = ch.transmit(bits)
        # at 50 dB BER ~ 10^-15, so 0 errors expected in 1000 bits
        assert stats.num_errors == 0

    def test_awgn_extreme_snr_no_crash(self):
        """Very high or very low SNR should not crash."""
        ch_high = AWGNChannel(snr_db=100.0)
        ch_low = AWGNChannel(snr_db=-100.0)
        assert 0 <= ch_high.equivalent_ber <= 0.5
        assert 0 <= ch_low.equivalent_ber <= 0.5

    def test_burst_rate_one_lots_of_errors(self):
        ch = BurstErrorChannel(burst_length=5, burst_rate=1.0, seed=0)
        bits = [0] * 100
        stats = ch.transmit(bits)
        # rate=1.0 hits a burst at every position checked
        assert stats.num_errors > 50

    def test_burst_empty_input(self):
        ch = BurstErrorChannel(burst_length=5, burst_rate=0.5, seed=0)
        stats = ch.transmit([])
        assert stats.received == []
        assert stats.num_errors == 0
