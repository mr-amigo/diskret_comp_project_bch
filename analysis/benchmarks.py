"""
Benchmark utilities and plotting.

Generates:
    - Success rate vs BER curves for multiple BCH codes (decoder reliability).
    - BER (output) vs BER (input) — coding gain visualization.
    - Encode/decode timing as a function of code parameters.
    - Comparison: BCH vs uncoded vs simple repetition code.

Saves figures as PNG. Uses matplotlib with a clean, publication-friendly style.
"""

from __future__ import annotations
import os
import random
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for batch runs
import matplotlib.pyplot as plt
import numpy as np

from core.bch import BCHCode
from core.channel import BinarySymmetricChannel, AWGNChannel


# ---------- styling ----------

PLOT_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "DejaVu Sans",
}

# Palette: muted, distinctive
PALETTE = ["#2E5EAA", "#E63946", "#F4A261", "#2A9D8F", "#7B2CBF", "#264653", "#E76F51"]


def setup_style():
    plt.rcParams.update(PLOT_STYLE)


# ---------- data structures ----------

@dataclass
class BenchmarkPoint:
    ber_in: float
    n_trials: int
    n_corrected: int
    n_failed: int
    n_miscorrected: int
    avg_input_errors: float
    avg_output_errors: float

    @property
    def success_rate(self) -> float:
        return self.n_corrected / self.n_trials

    @property
    def output_ber(self) -> float:
        # average residual BER on the message bits
        return self.avg_output_errors / max(1, self.n_trials)


# ---------- core benchmark loop ----------

def benchmark_bsc(
    code: BCHCode,
    bers: list[float],
    trials: int = 500,
    seed: int = 0,
) -> list[BenchmarkPoint]:
    rng = random.Random(seed)
    results = []
    for ber in bers:
        ch = BinarySymmetricChannel(p=ber, seed=rng.randint(0, 2**31))
        n_corrected = n_failed = n_misc = 0
        total_input_errors = 0
        total_output_errors = 0
        for _ in range(trials):
            msg = [rng.randint(0, 1) for _ in range(code.k)]
            cw = code.encode(msg)
            stats = ch.transmit(cw)
            total_input_errors += stats.num_errors
            res = code.decode(stats.received)
            if res.success and res.decoded_message == msg:
                n_corrected += 1
            elif not res.success:
                n_failed += 1
                # in failure case, decoder gives the unchanged received word -> count message-side errors
                total_output_errors += sum(
                    1 for a, b in zip(msg, stats.received[: code.k]) if a != b
                )
            else:
                n_misc += 1
                total_output_errors += sum(
                    1 for a, b in zip(msg, res.decoded_message) if a != b
                )
        results.append(
            BenchmarkPoint(
                ber_in=ber,
                n_trials=trials,
                n_corrected=n_corrected,
                n_failed=n_failed,
                n_miscorrected=n_misc,
                avg_input_errors=total_input_errors / trials,
                avg_output_errors=total_output_errors / (trials * code.k),
            )
        )
    return results


# ---------- plotting ----------

def plot_success_rate_vs_ber(
    codes: list[BCHCode],
    bers: list[float],
    trials: int,
    out_path: str,
    seed: int = 0,
):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, code in enumerate(codes):
        results = benchmark_bsc(code, bers, trials=trials, seed=seed + i)
        ys = [r.success_rate * 100 for r in results]
        ax.plot(
            bers, ys,
            marker="o", markersize=5,
            color=PALETTE[i % len(PALETTE)],
            label=f"BCH({code.n},{code.k},{code.t})",
            linewidth=2, alpha=0.9,
        )
    ax.set_xlabel("Channel BER")
    ax.set_ylabel("Decoding success rate (%)")
    ax.set_title("BCH performance on Binary Symmetric Channel")
    ax.set_ylim(-2, 105)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_coding_gain(
    code: BCHCode,
    bers: list[float],
    trials: int,
    out_path: str,
    seed: int = 0,
):
    """Compare output BER (after decoding) with input BER (uncoded)."""
    setup_style()
    results = benchmark_bsc(code, bers, trials=trials, seed=seed)
    out_bers = [max(r.output_ber, 1e-7) for r in results]  # floor for log plot

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bers, bers, "--", color="gray", label="Uncoded (y=x)", linewidth=1.5)
    ax.plot(bers, out_bers, marker="s", color=PALETTE[1],
            label=f"After BCH({code.n},{code.k},{code.t})",
            linewidth=2, markersize=5)
    ax.set_xlabel("Input BER (channel)")
    ax.set_ylabel("Output BER (after decode)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title(f"Coding gain — BCH({code.n},{code.k},{code.t})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timing(
    parameters: list[tuple[int, int]],  # list of (m, t)
    n_messages: int,
    out_path: str,
    seed: int = 0,
):
    setup_style()
    rng = random.Random(seed)
    encode_times = []
    decode_times = []
    decode_with_err_times = []
    labels = []

    for m, t in parameters:
        code = BCHCode(m=m, t=t)
        labels.append(f"({code.n},{code.k},{t})")
        # encode timing
        t0 = time.perf_counter()
        for _ in range(n_messages):
            msg = [rng.randint(0, 1) for _ in range(code.k)]
            code.encode(msg)
        encode_times.append((time.perf_counter() - t0) / n_messages * 1000)
        # decode no-error timing
        msg = [rng.randint(0, 1) for _ in range(code.k)]
        cw = code.encode(msg)
        t0 = time.perf_counter()
        for _ in range(n_messages):
            code.decode(cw)
        decode_times.append((time.perf_counter() - t0) / n_messages * 1000)
        # decode with t errors timing
        positions = rng.sample(range(code.n), t)
        received = list(cw)
        for p in positions:
            received[p] ^= 1
        t0 = time.perf_counter()
        for _ in range(n_messages):
            code.decode(received)
        decode_with_err_times.append((time.perf_counter() - t0) / n_messages * 1000)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(parameters))
    w = 0.27
    ax.bar(x - w, encode_times, width=w, label="Encode", color=PALETTE[0])
    ax.bar(x, decode_times, width=w, label="Decode (0 errors)", color=PALETTE[3])
    ax.bar(x + w, decode_with_err_times, width=w, label="Decode (t errors)", color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Time per operation (ms)")
    ax.set_title("Encode / Decode timing across BCH codes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bch_vs_repetition(
    code: BCHCode,
    repetition: int,
    bers: list[float],
    trials: int,
    out_path: str,
    seed: int = 0,
):
    """Compare BCH vs majority-vote repetition code with the same overall rate."""
    setup_style()

    # BCH points
    bch_points = benchmark_bsc(code, bers, trials=trials, seed=seed)
    bch_residual = [max(r.output_ber, 1e-7) for r in bch_points]

    # Repetition (n=repetition, k=1): majority vote.
    # Probability of bit error after decoding for repetition n with input BER p:
    #   P_err = sum_{j > n/2} C(n,j) p^j (1-p)^(n-j)
    rep_residual = []
    for p in bers:
        n = repetition
        Pe = 0.0
        from math import comb
        for j in range((n // 2) + 1, n + 1):
            Pe += comb(n, j) * (p ** j) * ((1 - p) ** (n - j))
        rep_residual.append(max(Pe, 1e-7))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bers, bers, "--", color="gray", label="Uncoded", linewidth=1.5)
    ax.plot(bers, bch_residual, marker="s", color=PALETTE[0],
            label=f"BCH({code.n},{code.k},{code.t})  rate={code.code_rate:.2f}",
            linewidth=2)
    ax.plot(bers, rep_residual, marker="^", color=PALETTE[2],
            label=f"Repetition n={repetition}  rate={1/repetition:.2f}",
            linewidth=2)
    ax.set_xlabel("Channel BER")
    ax.set_ylabel("Residual BER")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title("BCH vs Repetition code (residual BER)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_awgn_performance(
    code: BCHCode,
    snrs_db: list[float],
    trials: int,
    out_path: str,
    seed: int = 0,
):
    """BER vs SNR over AWGN channel with hard-decision BPSK."""
    setup_style()
    rng = random.Random(seed)
    uncoded_ber = []
    coded_ber = []
    for snr in snrs_db:
        ch = AWGNChannel(snr_db=snr, seed=rng.randint(0, 2**31))
        uncoded_ber.append(ch.equivalent_ber)
        # measure coded performance
        n_total_bit_errors = 0
        n_total_bits = 0
        for _ in range(trials):
            msg = [rng.randint(0, 1) for _ in range(code.k)]
            cw = code.encode(msg)
            stats = ch.transmit(cw)
            res = code.decode(stats.received)
            if res.success:
                n_total_bit_errors += sum(
                    1 for a, b in zip(msg, res.decoded_message) if a != b
                )
            else:
                n_total_bit_errors += sum(
                    1 for a, b in zip(msg, stats.received[: code.k]) if a != b
                )
            n_total_bits += code.k
        coded_ber.append(max(n_total_bit_errors / n_total_bits, 1e-7))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(snrs_db, uncoded_ber, "--", color="gray",
                label="Uncoded (theoretical Q-function)", linewidth=1.5)
    ax.semilogy(snrs_db, coded_ber, marker="o", color=PALETTE[0],
                label=f"BCH({code.n},{code.k},{code.t}) (simulated)",
                linewidth=2, markersize=6)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title(f"AWGN performance: BCH({code.n},{code.k},{code.t}) vs uncoded")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- runner ----------

def run_full_suite(out_dir: str = "examples/plots") -> None:
    """Generate all standard plots and save them."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing plots to {out_dir}/")

    print("[1/5] Success rate vs BER for several codes...")
    bers = [round(0.005 + 0.02 * i, 4) for i in range(20)]
    codes = [
        BCHCode(m=5, t=1),
        BCHCode(m=5, t=3),
        BCHCode(m=5, t=5),
        BCHCode(m=6, t=3),
        BCHCode(m=7, t=4),
    ]
    plot_success_rate_vs_ber(codes, bers, trials=200,
                              out_path=os.path.join(out_dir, "success_vs_ber.png"))

    print("[2/5] Coding gain (residual BER)...")
    bers = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]
    plot_coding_gain(BCHCode(m=6, t=3), bers, trials=300,
                     out_path=os.path.join(out_dir, "coding_gain.png"))

    print("[3/5] Encode/Decode timing...")
    plot_timing(
        parameters=[(4, 2), (5, 3), (6, 3), (7, 4), (8, 5)],
        n_messages=200,
        out_path=os.path.join(out_dir, "timing.png"),
    )

    print("[4/5] BCH vs repetition...")
    bers = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]
    plot_bch_vs_repetition(BCHCode(m=5, t=3), repetition=3, bers=bers,
                            trials=300,
                            out_path=os.path.join(out_dir, "bch_vs_repetition.png"))

    print("[5/5] AWGN performance...")
    snrs = list(range(0, 10))
    plot_awgn_performance(BCHCode(m=6, t=3), snrs, trials=200,
                          out_path=os.path.join(out_dir, "awgn_ber.png"))

    print("Done.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run benchmark suite and plot results.")
    ap.add_argument("--out", default="examples/plots", help="Output directory")
    args = ap.parse_args()
    run_full_suite(args.out)
