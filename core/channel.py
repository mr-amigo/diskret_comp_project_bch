"""
Channel models for simulating bit errors during transmission.

A "channel" takes a transmitted bit sequence and returns a possibly-corrupted
received sequence. Three models are provided:

1. BinarySymmetricChannel (BSC): each bit is flipped independently with
   probability p. Standard model for memoryless channels.

2. BurstErrorChannel: errors occur in bursts of length L at rate lambda.
   Models impulsive noise, scratches on disks, deep fades.

3. AWGNChannel: Additive White Gaussian Noise. Maps bits to BPSK symbols
   {-1, +1}, adds Gaussian noise of variance sigma^2, then hard-decides
   to recover bits. Configured by SNR (in dB).
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import random
from typing import Protocol, Sequence, runtime_checkable


@dataclass
class ChannelStats:
    """Statistics about a channel realization."""
    transmitted: list[int]
    received: list[int]
    error_positions: list[int]

    @property
    def num_errors(self) -> int:
        return len(self.error_positions)

    @property
    def empirical_ber(self) -> float:
        return self.num_errors / max(1, len(self.transmitted))


@runtime_checkable
class Channel(Protocol):
    """Protocol shared by all channel models — anything with `transmit(bits)`."""
    def transmit(self, bits: Sequence[int]) -> ChannelStats: ...


class BinarySymmetricChannel:
    """BSC(p): each bit independently flipped with probability p."""

    def __init__(self, p: float, seed: int | None = None):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p
        self.rng = random.Random(seed)

    def transmit(self, bits: Sequence[int]) -> ChannelStats:
        received = []
        errors = []
        for i, b in enumerate(bits):
            if self.rng.random() < self.p:
                received.append(1 - b)
                errors.append(i)
            else:
                received.append(b)
        return ChannelStats(list(bits), received, errors)

    def transmit_with_exact_errors(
        self, bits: Sequence[int], num_errors: int
    ) -> ChannelStats:
        """Inject exactly `num_errors` bit flips at uniformly random positions."""
        if num_errors > len(bits):
            raise ValueError("num_errors > length")
        positions = self.rng.sample(range(len(bits)), num_errors)
        positions.sort()
        received = list(bits)
        for p in positions:
            received[p] ^= 1
        return ChannelStats(list(bits), received, positions)

    def __repr__(self) -> str:
        return f"BSC(p={self.p})"


class BurstErrorChannel:
    """
    Channel that injects bursts of errors.

    A burst is a contiguous run of bit positions where errors are placed
    with probability `burst_density`. Bursts start at independent positions
    with rate `burst_rate` (expected number of bursts per bit).
    """

    def __init__(
        self,
        burst_length: int,
        burst_rate: float,
        burst_density: float = 1.0,
        seed: int | None = None,
    ):
        if burst_length < 1:
            raise ValueError("burst_length must be >= 1")
        if not (0.0 <= burst_rate <= 1.0):
            raise ValueError("burst_rate must be in [0, 1]")
        if not (0.0 <= burst_density <= 1.0):
            raise ValueError("burst_density must be in [0, 1]")
        self.burst_length = burst_length
        self.burst_rate = burst_rate
        self.burst_density = burst_density
        self.rng = random.Random(seed)

    def transmit(self, bits: Sequence[int]) -> ChannelStats:
        received = list(bits)
        errors: list[int] = []
        n = len(bits)
        i = 0
        while i < n:
            if self.rng.random() < self.burst_rate:
                # start a burst
                end = min(i + self.burst_length, n)
                for j in range(i, end):
                    if self.rng.random() < self.burst_density:
                        received[j] ^= 1
                        errors.append(j)
                i = end
            else:
                i += 1
        return ChannelStats(list(bits), received, errors)

    def transmit_one_burst(
        self, bits: Sequence[int], position: int | None = None
    ) -> ChannelStats:
        """Inject a single burst of length self.burst_length starting at `position`."""
        n = len(bits)
        if position is None:
            position = self.rng.randint(0, n - 1)
        received = list(bits)
        errors = []
        for j in range(position, min(position + self.burst_length, n)):
            received[j] ^= 1
            errors.append(j)
        return ChannelStats(list(bits), received, errors)

    def __repr__(self) -> str:
        return (
            f"BurstChannel(L={self.burst_length}, "
            f"rate={self.burst_rate}, density={self.burst_density})"
        )


class AWGNChannel:
    """
    Additive White Gaussian Noise channel with BPSK modulation and hard decision.

    Bit b in {0, 1} is mapped to symbol s = 1 - 2*b in {+1, -1}.
    Received symbol y = s + noise, noise ~ Normal(0, sigma^2).
    Hard decision: b_hat = 0 if y >= 0 else 1.

    Effective bit-flip probability:
        p = Q(1/sigma) = 0.5 * erfc(1 / (sigma * sqrt(2)))
    where Q is the tail of the standard normal.

    SNR in dB is defined as 10 * log10(Es / N0) where Es = 1 (signal energy
    per symbol) and N0 = 2 * sigma^2. Equivalently, sigma^2 = 0.5 * 10^(-SNR/10).
    """

    def __init__(self, snr_db: float, seed: int | None = None):
        self.snr_db = snr_db
        self.sigma = math.sqrt(0.5 * 10 ** (-snr_db / 10))
        self.rng = random.Random(seed)

    @property
    def equivalent_ber(self) -> float:
        """Probability that a transmitted bit is flipped after hard decision."""
        return 0.5 * math.erfc(1.0 / (self.sigma * math.sqrt(2)))

    def transmit(self, bits: Sequence[int]) -> ChannelStats:
        received: list[int] = []
        errors: list[int] = []
        for i, b in enumerate(bits):
            s = 1.0 - 2.0 * b
            y = s + self.rng.gauss(0.0, self.sigma)
            b_hat = 0 if y >= 0 else 1
            received.append(b_hat)
            if b_hat != b:
                errors.append(i)
        return ChannelStats(list(bits), received, errors)

    def __repr__(self) -> str:
        return f"AWGN(SNR={self.snr_db}dB, sigma={self.sigma:.4f})"
