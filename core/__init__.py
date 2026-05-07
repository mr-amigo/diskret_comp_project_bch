"""Core mathematical components for the BCH error correction simulator."""

from .galois import GaloisField
from .polynomial import Polynomial
from .bch import BCHCode, DecodeResult, BMStep
from .channel import (
    Channel,
    BinarySymmetricChannel,
    BurstErrorChannel,
    AWGNChannel,
    ChannelStats,
)

__all__ = [
    "GaloisField",
    "Polynomial",
    "BCHCode",
    "DecodeResult",
    "BMStep",
    "Channel",
    "BinarySymmetricChannel",
    "BurstErrorChannel",
    "AWGNChannel",
    "ChannelStats",
]
