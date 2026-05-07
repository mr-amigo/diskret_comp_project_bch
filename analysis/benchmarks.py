"""Generate success rate vs BER plot for several BCH codes."""

import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.bch import BCHCode
from core.channel import BinarySymmetricChannel


def benchmark(code: BCHCode, bers: list[float], trials: int, seed: int = 0) -> list[float]:
    rates = []
    for ber in bers:
        ch = BinarySymmetricChannel(p=ber, seed=seed)
        rng = random.Random(seed)
        ok = 0
        for _ in range(trials):
            msg = [rng.randint(0, 1) for _ in range(code.k)]
            cw = code.encode(msg)
            received, _ = ch.transmit(cw)
            res = code.decode(received)
            if res.success and res.decoded_message == msg:
                ok += 1
        rates.append(ok / trials)
    return rates


def main(out_dir: str = "examples/plots") -> None:
    os.makedirs(out_dir, exist_ok=True)

    bers = [round(0.005 + 0.02 * i, 4) for i in range(20)]
    codes = [
        BCHCode(m=5, t=1),
        BCHCode(m=5, t=3),
        BCHCode(m=5, t=5),
        BCHCode(m=6, t=3),
        BCHCode(m=7, t=4),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, code in enumerate(codes):
        rates = benchmark(code, bers, trials=200, seed=i)
        ys = [r * 100 for r in rates]
        ax.plot(
            bers, ys,
            marker="o", markersize=4,
            label=f"BCH({code.n},{code.k},{code.t})",
            linewidth=2,
        )
    ax.set_xlabel("Channel BER")
    ax.set_ylabel("Decoding success rate (%)")
    ax.set_title("BCH performance on Binary Symmetric Channel")
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, "success_vs_ber.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
