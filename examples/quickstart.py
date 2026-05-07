"""
Quick-start example: encode, corrupt, decode.

Run: python examples/quickstart.py
"""

import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.bch import BCHCode
from core.channel import BinarySymmetricChannel


def main():
    # 1. Build a BCH(31, 16, 3) code
    code = BCHCode(m=5, t=3)
    print(f"Code: {code}")
    print(f"  rate = k/n = {code.code_rate:.3f}")
    print(f"  generator: {code.generator}")
    print()

    # 2. Encode a random message
    rng = random.Random(42)
    message = [rng.randint(0, 1) for _ in range(code.k)]
    codeword = code.encode(message)
    print(f"Message  ({code.k} bits): {''.join(map(str, message))}")
    print(f"Codeword ({code.n} bits): {''.join(map(str, codeword))}")
    print()

    # 3. Send through a noisy channel — pick BER such that errors ≤ t are likely
    channel = BinarySymmetricChannel(p=0.05, seed=7)
    stats = channel.transmit(codeword)
    print(f"Channel: BSC(p=0.05), errors injected: {stats.num_errors}")
    print(f"Received:                {''.join(map(str, stats.received))}")
    print()

    # 4. Decode
    result = code.decode(stats.received)
    if result.success:
        print(f"✓ Decoder corrected {result.num_errors_corrected} error(s)")
        print(f"  at positions (MSB-first): "
              f"{sorted(result.error_positions_msb_first(code.n))}")
        print(f"  recovered message: {''.join(map(str, result.decoded_message))}")
        print(f"  matches original: {result.decoded_message == message}")
    else:
        print(f"✗ Decoder failed (more than t = {code.t} errors)")


if __name__ == "__main__":
    main()
