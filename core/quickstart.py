"""Minimal example: encode a message, corrupt it, decode it back."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.bch import BCHCode


bch = BCHCode(m=5, t=3)
print(f"Code: BCH(n={bch.n}, k={bch.k}, t={bch.t})")

message = [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
print(f"Message:    {''.join(map(str, message))}")

codeword = bch.encode(message)
print(f"Codeword:   {''.join(map(str, codeword))}")

# Inject 3 errors manually
received = list(codeword)
for pos in [2, 14, 27]:
    received[pos] ^= 1
print(f"Received:   {''.join(map(str, received))}")
print(f"3 errors at positions [2, 14, 27]")

result = bch.decode(received)
print(f"Decoded:    {''.join(map(str, result.decoded_codeword))}")
print(f"Errors corrected at: {result.error_positions}")

if result.decoded_message == message:
    print("Original message recovered.")
else:
    print("Decoding failed.")
