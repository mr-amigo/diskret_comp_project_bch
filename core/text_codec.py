"""Text encoding/decoding via BCH codes."""


def text_to_bits(text):
    """Convert text to bits (UTF-8, MSB-first)."""
    data = text.encode('utf-8')
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_text(bits):
    """Convert bits back to text. Returns text with replacement chars for invalid bytes."""
    n_bytes = len(bits) // 8
    data = bytearray()
    for byte_idx in range(n_bytes):
        byte = 0
        for i in range(8):
            byte = (byte << 1) | bits[byte_idx * 8 + i]
        data.append(byte)
    return data.decode('utf-8', errors='replace')


def encode_text(text, bch):
    """Encode text into list of BCH codewords. Returns (codewords, original_bit_length)."""
    bits = text_to_bits(text)
    original_length = len(bits)
    padding = (-len(bits)) % bch.k
    bits.extend([0] * padding)

    codewords = []
    for i in range(0, len(bits), bch.k):
        codewords.append(bch.encode(bits[i:i + bch.k]))
    return codewords, original_length


def decode_text(codewords, bch, original_length):
    """Decode codewords back to text. Returns (text, n_success, n_failed)."""
    all_bits = []
    n_ok = 0
    n_fail = 0
    for cw in codewords:
        result = bch.decode(cw)
        if result.success:
            n_ok += 1
            all_bits.extend(result.decoded_message)
        else:
            n_fail += 1
            all_bits.extend(cw[:bch.k])
    all_bits = all_bits[:original_length]
    return bits_to_text(all_bits), n_ok, n_fail
