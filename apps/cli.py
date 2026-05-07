"""Command-line interface for the BCH simulator."""

import argparse
import random
import sys

from core.bch import BCHCode
from core.channel import BinarySymmetricChannel


def cmd_encode(args):
    bch = BCHCode(m=args.m, t=args.t)
    message = [int(c) for c in args.message if c in "01"]
    if len(message) != bch.k:
        print(f"Error: message length must be {bch.k}, got {len(message)}")
        return 1
    codeword = bch.encode(message)
    print(f"Code:     BCH(n={bch.n}, k={bch.k}, t={bch.t})")
    print(f"Message:  {''.join(map(str, message))}")
    print(f"Codeword: {''.join(map(str, codeword))}")
    return 0


def cmd_decode(args):
    bch = BCHCode(m=args.m, t=args.t)
    received = [int(c) for c in args.received if c in "01"]
    if len(received) != bch.n:
        print(f"Error: received length must be {bch.n}, got {len(received)}")
        return 1
    result = bch.decode(received)
    print(f"Code:     BCH(n={bch.n}, k={bch.k}, t={bch.t})")
    print(f"Received: {''.join(map(str, received))}")
    if result.success:
        print(f"Decoded:  {''.join(map(str, result.decoded_codeword))}")
        print(f"Message:  {''.join(map(str, result.decoded_message))}")
        if result.error_positions:
            print(f"Corrected {len(result.error_positions)} error(s) at positions {result.error_positions}")
        else:
            print("No errors detected")
    else:
        print("Decoding failed: too many errors (>t)")
    return 0


def cmd_simulate(args):
    bch = BCHCode(m=args.m, t=args.t)
    rng = random.Random(args.seed)
    message = [rng.randint(0, 1) for _ in range(bch.k)]
    codeword = bch.encode(message)
    channel = BinarySymmetricChannel(p=args.ber, seed=args.seed)
    received, error_positions = channel.transmit(codeword)
    result = bch.decode(received)

    print(f"Code:     BCH(n={bch.n}, k={bch.k}, t={bch.t})")
    print(f"BER:      {args.ber}")
    print()
    print(f"Message:  {''.join(map(str, message))}")
    print(f"Codeword: {''.join(map(str, codeword))}")
    print(f"Received: {''.join(map(str, received))}")
    print(f"Errors:   {len(error_positions)} at positions {error_positions}")
    print()
    if result.success:
        print(f"Decoded:  {''.join(map(str, result.decoded_codeword))}")
        if result.decoded_message == message:
            print("Result:   SUCCESS — original message recovered")
        else:
            print("Result:   MISCORRECTION — wrong message decoded")
    else:
        print("Result:   FAILED — too many errors")
    return 0


def cmd_plot(args):
    import matplotlib.pyplot as plt

    bch = BCHCode(m=args.m, t=args.t)
    bers = [i * args.max_ber / args.steps for i in range(1, args.steps + 1)]
    success_rates = []

    for ber in bers:
        rng = random.Random(args.seed)
        ch = BinarySymmetricChannel(p=ber, seed=args.seed)
        ok = 0
        for _ in range(args.trials):
            msg = [rng.randint(0, 1) for _ in range(bch.k)]
            cw = bch.encode(msg)
            received, _ = ch.transmit(cw)
            result = bch.decode(received)
            if result.success and result.decoded_message == msg:
                ok += 1
        rate = ok / args.trials
        success_rates.append(rate)
        print(f"  BER={ber:.3f}: success rate = {rate:.2%}")

    plt.figure(figsize=(8, 5))
    plt.plot(bers, success_rates, marker="o")
    plt.xlabel("Bit Error Rate (BER)")
    plt.ylabel("Success rate")
    plt.title(f"BCH(n={bch.n}, k={bch.k}, t={bch.t}) — success rate vs BER")
    plt.grid(True)
    plt.savefig(args.output)
    print(f"\nPlot saved to {args.output}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="BCH error-correction simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_enc = sub.add_parser("encode", help="Encode a message")
    p_enc.add_argument("--m", type=int, required=True)
    p_enc.add_argument("--t", type=int, required=True)
    p_enc.add_argument("--message", required=True)
    p_enc.set_defaults(func=cmd_encode)

    p_dec = sub.add_parser("decode", help="Decode a received vector")
    p_dec.add_argument("--m", type=int, required=True)
    p_dec.add_argument("--t", type=int, required=True)
    p_dec.add_argument("--received", required=True)
    p_dec.set_defaults(func=cmd_decode)

    p_sim = sub.add_parser("simulate", help="Run encode -> channel -> decode")
    p_sim.add_argument("--m", type=int, required=True)
    p_sim.add_argument("--t", type=int, required=True)
    p_sim.add_argument("--ber", type=float, default=0.05)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.set_defaults(func=cmd_simulate)

    p_plot = sub.add_parser("plot", help="Plot success rate vs BER")
    p_plot.add_argument("--m", type=int, required=True)
    p_plot.add_argument("--t", type=int, required=True)
    p_plot.add_argument("--max-ber", type=float, default=0.3)
    p_plot.add_argument("--steps", type=int, default=15)
    p_plot.add_argument("--trials", type=int, default=200)
    p_plot.add_argument("--seed", type=int, default=0)
    p_plot.add_argument("--output", default="success_vs_ber.png")
    p_plot.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
