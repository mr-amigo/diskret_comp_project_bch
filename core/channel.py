"""Binary symmetric channel: each bit independently flipped with probability p."""

import random
from typing import Sequence


class BinarySymmetricChannel:
    def __init__(self, p: float, seed: int | None = None):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p
        self.rng = random.Random(seed)

    def transmit(self, bits: Sequence[int]) -> tuple[list[int], list[int]]:
        received = []
        errors = []
        for i, b in enumerate(bits):
            if self.rng.random() < self.p:
                received.append(1 - b)
                errors.append(i)
            else:
                received.append(b)
        return received, errors
