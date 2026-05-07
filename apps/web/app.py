"""Streamlit dashboard for BCH simulator."""

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import matplotlib.pyplot as plt

from core.bch import BCHCode
from core.channel import BinarySymmetricChannel


st.set_page_config(page_title="BCH Simulator", layout="wide")
st.title("BCH Error-Correction Simulator")
st.caption("Курсовий проєкт з дискретної математики, УКУ")

st.sidebar.header("Code parameters")
m = st.sidebar.slider("m (field size 2^m)", 3, 8, 5)
t = st.sidebar.slider("t (errors corrected)", 1, 7, 3)

try:
    code = BCHCode(m=m, t=t)
except ValueError as e:
    st.sidebar.error(str(e))
    st.stop()

st.sidebar.write(f"**n** = {code.n}")
st.sidebar.write(f"**k** = {code.k}")
st.sidebar.write(f"**rate** = {code.code_rate:.3f}")

tab1, tab2, tab3 = st.tabs(["Encoder/Decoder", "Channel Simulator", "Performance"])


with tab1:
    st.header("Encode and decode messages")

    if "msg_bits" not in st.session_state or len(st.session_state.msg_bits) != code.k:
        random.seed(42)
        st.session_state.msg_bits = [random.randint(0, 1) for _ in range(code.k)]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Message")
        msg_str = st.text_input(
            f"Bit string of length {code.k}",
            value="".join(map(str, st.session_state.msg_bits)),
            max_chars=code.k,
        )
        msg_str = "".join(c for c in msg_str if c in "01")
        if len(msg_str) != code.k:
            msg_str = (msg_str + "0" * code.k)[:code.k]
        st.session_state.msg_bits = [int(c) for c in msg_str]

        if st.button("Random message"):
            st.session_state.msg_bits = [random.randint(0, 1) for _ in range(code.k)]
            st.rerun()

    with col2:
        st.subheader("Codeword")
        codeword = code.encode(st.session_state.msg_bits)
        st.code("".join(map(str, codeword)))
        st.write(f"Length: {len(codeword)} bits ({code.k} message + {code.n - code.k} parity)")

    st.divider()
    st.subheader("Inject errors")

    n_errors = st.slider("Number of errors to inject", 0, code.n, min(code.t, 2))
    error_seed = st.number_input("Error position seed", value=0, step=1)

    rng = random.Random(error_seed)
    error_positions = sorted(rng.sample(range(code.n), n_errors)) if n_errors else []
    received = list(codeword)
    for p in error_positions:
        received[p] ^= 1

    col3, col4 = st.columns(2)
    with col3:
        st.text("Received (with errors):")
        st.code("".join(map(str, received)))
        st.write(f"Errors at positions: {error_positions}")
    with col4:
        result = code.decode(received)
        st.text("Decoder output:")
        st.code("".join(map(str, result.decoded_codeword)))
        if result.success and result.decoded_message == st.session_state.msg_bits:
            st.success(f"Recovered original message ({len(result.error_positions)} errors corrected)")
        elif result.success:
            st.warning("Decoder converged but to a wrong codeword (miscorrection)")
        else:
            st.error("Decoder failed: too many errors")


with tab2:
    st.header("Channel simulation")
    st.write("Run multiple transmissions through a binary symmetric channel.")

    col1, col2 = st.columns(2)
    with col1:
        ber = st.slider("Bit error rate p", 0.0, 0.5, 0.05, 0.005)
    with col2:
        n_messages = st.slider("Number of messages", 1, 500, 50)

    if st.button("Run transmission"):
        rng = random.Random(0)
        ch = BinarySymmetricChannel(p=ber, seed=0)

        n_success = 0
        total_errors = 0
        total_corrected = 0

        for _ in range(n_messages):
            msg = [rng.randint(0, 1) for _ in range(code.k)]
            cw = code.encode(msg)
            received, error_positions = ch.transmit(cw)
            total_errors += len(error_positions)
            result = code.decode(received)
            if result.success and result.decoded_message == msg:
                n_success += 1
                total_corrected += len(result.error_positions)

        col1, col2, col3 = st.columns(3)
        col1.metric("Success rate", f"{n_success / n_messages:.1%}")
        col2.metric("Errors injected", total_errors)
        col3.metric("Errors corrected", total_corrected)


with tab3:
    st.header("Success rate vs BER")
    st.write("How well does the code perform across different noise levels?")

    col1, col2 = st.columns(2)
    with col1:
        max_ber = st.slider("Maximum BER", 0.05, 0.5, 0.3, 0.05)
        n_steps = st.slider("Number of points", 5, 30, 15)
    with col2:
        n_trials = st.slider("Trials per point", 50, 500, 200)

    if st.button("Generate plot"):
        bers = [i * max_ber / n_steps for i in range(1, n_steps + 1)]
        rates = []
        progress = st.progress(0.0)

        for idx, p in enumerate(bers):
            ch = BinarySymmetricChannel(p=p, seed=idx)
            rng = random.Random(idx)
            ok = 0
            for _ in range(n_trials):
                msg = [rng.randint(0, 1) for _ in range(code.k)]
                cw = code.encode(msg)
                received, _ = ch.transmit(cw)
                result = code.decode(received)
                if result.success and result.decoded_message == msg:
                    ok += 1
            rates.append(ok / n_trials)
            progress.progress((idx + 1) / len(bers))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(bers, rates, marker="o")
        ax.set_xlabel("Bit Error Rate (BER)")
        ax.set_ylabel("Success rate")
        ax.set_title(f"BCH(n={code.n}, k={code.k}, t={code.t})")
        ax.grid(True)
        ax.set_ylim(0, 1.05)
        st.pyplot(fig)
