"""Block-DCT frequency-domain watermark embedding and extraction.

Scheme overview
---------------
* The image is converted to grayscale and resized to a fixed square canvas.
* The payload (watermark + version + digest + CRC) is protected by a
  Reed-Solomon(223, 81) code and prefixed with a 32-bit synchronisation
  word.  Every frame bit is embedded ``REPETITION`` times across the
  (3, 2) and (2, 3) mid-frequency DCT coefficients of pseudo-randomly
  permuted 8x8 blocks, using additive spread-spectrum modulation with a
  content-adaptive amplitude.  A weak pilot tone (orthogonal to the
  payload coefficients) is added for geometric synchronisation.
* Extraction estimates the geometric scale from the pilot's FFT peak,
  refines scale + translation by maximising the frame energy, verifies
  the sync word, then soft-combines the repeated bits and RS-decodes
  the payload.
"""

import math
import struct
import zlib

import numpy as np
from PIL import Image

from .rs import rs_encode, rs_decode, RSDecodeError

# ---------------------------------------------------------------------------
# Codec parameters (stored per-asset so extraction stays self-describing)
# ---------------------------------------------------------------------------

CANVAS_SIZE = 1024          # fixed square canvas
BLOCK = 8                   # DCT block edge
COEFS = ((3, 2), (2, 3))    # mid-frequency coefficient pair (row, col)
ALPHA = 32.0                # minimum additive embedding amplitude
ALPHA_MAX = 64.0            # cap for content-adaptive strengthening
ALPHA_ADAPT = 1.35          # alpha >= ALPHA_ADAPT * content coefficient std
MAX_WATERMARK_BYTES = 64
RS_NSYM = 142               # RS(223, 81): corrects up to 71 byte errors
REPETITION = 18             # each frame bit embedded in this many slots
PERM_SEED = 20240613
SYNC_SEED = 999

# Pilot tone: a weak pure cosine whose FFT peak reveals the geometric
# scale of a received image.  Its frequency matches DCT bin (1, 2) of the
# 8x8 grid, i.e. it is orthogonal to both payload coefficients.
PILOT_FX = 2.0 / 8.0        # cycles per pixel (canvas coordinates)
PILOT_FY = 1.0 / 8.0
PILOT_AMP = 8.0
PILOT_MIN_PROMINENCE = 4.0

_SEARCH_REGION = 512        # centre canvas region used for geometry search

PAYLOAD_LEN = 1 + MAX_WATERMARK_BYTES + 4 + 8 + 4   # 81 bytes
CODEWORD_BITS = (PAYLOAD_LEN + RS_NSYM) * 8          # 1784 bits
SYNC_BITS = 32
FRAME_BITS = SYNC_BITS + CODEWORD_BITS               # 1816 bits

BLOCKS_PER_SIDE = CANVAS_SIZE // BLOCK               # 128
NUM_BLOCKS = BLOCKS_PER_SIDE * BLOCKS_PER_SIDE       # 16384
NUM_SLOTS = NUM_BLOCKS * len(COEFS)                  # 32768

assert FRAME_BITS * REPETITION <= NUM_SLOTS

# Deterministic slot permutation and sync word (shared by embed/extract).
_PERM = np.random.RandomState(PERM_SEED).permutation(NUM_SLOTS)
SYNC_WORD = tuple(int(b) for b in np.random.RandomState(SYNC_SEED).randint(0, 2, SYNC_BITS))
_SYNC_SIGNS = np.array([1.0 if b else -1.0 for b in SYNC_WORD], dtype=np.float64)
_SYNC_SLOTS = _PERM[:SYNC_BITS * REPETITION]

# 8x8 orthonormal DCT-II basis.
_DCT_MAT = np.array(
    [[math.cos((2 * x + 1) * k * math.pi / 16) for x in range(8)] for k in range(8)],
    dtype=np.float64,
)
_DCT_MAT[0] *= math.sqrt(1.0 / 8.0)
_DCT_MAT[1:] *= math.sqrt(2.0 / 8.0)

# For each payload coefficient (row, col) = (y-frequency, x-frequency), the
# flattened 64-tap kernel extracts that coefficient from a flattened 8x8
# block; the same vector is the spatial basis image used for embedding.
_KERNELS = [np.kron(_DCT_MAT[r], _DCT_MAT[c]) for (r, c) in COEFS]


# ---------------------------------------------------------------------------
# Payload packing
# ---------------------------------------------------------------------------

class PayloadError(ValueError):
    pass


def pack_payload(watermark, version, digest8):
    """Pack watermark/version/digest into the 81-byte protected payload."""
    wb = watermark.encode("utf-8")
    if len(wb) > MAX_WATERMARK_BYTES:
        raise PayloadError(
            f"watermark is {len(wb)} bytes UTF-8, max {MAX_WATERMARK_BYTES}")
    if not (1 <= int(version) <= 0x7FFFFFFF):
        raise PayloadError("version must be a positive integer (< 2^31)")
    body = bytes([len(wb)]) + wb.ljust(MAX_WATERMARK_BYTES, b"\x00")
    body += struct.pack(">I", int(version))
    body += digest8
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + struct.pack(">I", crc)


def unpack_payload(payload):
    """Validate CRC and split a decoded payload; raises PayloadError."""
    if len(payload) != PAYLOAD_LEN:
        raise PayloadError("bad payload length")
    body, crc_bytes = payload[:-4], payload[-4:]
    if struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF) != crc_bytes:
        raise PayloadError("payload CRC mismatch")
    wlen = body[0]
    if wlen > MAX_WATERMARK_BYTES:
        raise PayloadError("corrupt watermark length")
    watermark = body[1:1 + wlen].decode("utf-8", errors="replace")
    version = struct.unpack(">I", body[65:69])[0]
    digest8 = body[69:77]
    return watermark, version, digest8


def _bytes_to_bits(data):
    return [(byte >> (7 - i)) & 1 for byte in data for i in range(8)]


def _bits_to_bytes(bits):
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i + 8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out)


# ---------------------------------------------------------------------------
# DCT helpers
# ---------------------------------------------------------------------------

def _blocks_of(gray):
    """gray: (CANVAS, CANVAS) array -> (NUM_BLOCKS, 64) flattened blocks."""
    n = BLOCKS_PER_SIDE
    return (gray.reshape(n, BLOCK, n, BLOCK)
            .transpose(0, 2, 1, 3)
            .reshape(NUM_BLOCKS, BLOCK * BLOCK))


def _slot_values(gray):
    """Value of every embedding slot: both DCT coefficients of every
    block, laid out as [block0_coef0, block0_coef1, block1_coef0, ...]."""
    blocks = _blocks_of(gray)
    out = np.empty(NUM_SLOTS, dtype=np.float64)
    out[0::2] = blocks @ _KERNELS[0]
    out[1::2] = blocks @ _KERNELS[1]
    return out


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def choose_alpha(gray):
    """Content-adaptive embedding amplitude.

    Noisy / textured content masks a stronger watermark, so the amplitude
    scales with the content's mid-frequency coefficient energy (clamped
    to [ALPHA, ALPHA_MAX]).  The decoder is sign-based and does not need
    to know the amplitude.
    """
    blocks = _blocks_of(np.asarray(gray, dtype=np.float64))
    std = 0.5 * (float((blocks @ _KERNELS[0]).std()) +
                 float((blocks @ _KERNELS[1]).std()))
    return float(min(ALPHA_MAX, max(ALPHA, ALPHA_ADAPT * std)))


def embed(watermark, version, digest8, image_gray):
    """Embed the payload into a PIL grayscale image of any size.

    Returns (watermarked PIL image "L" CANVAS_SIZE x CANVAS_SIZE, alpha).
    """
    canvas = image_gray.convert("L").resize(
        (CANVAS_SIZE, CANVAS_SIZE), Image.BICUBIC)
    gray = np.asarray(canvas, dtype=np.float64)
    alpha = choose_alpha(gray)

    payload = pack_payload(watermark, version, digest8)
    codeword = rs_encode(list(payload), RS_NSYM)
    frame_bits = list(SYNC_WORD) + _bytes_to_bits(codeword)

    # Embedding +-alpha on a DCT coefficient is equivalent to adding the
    # corresponding basis image in the spatial domain.  Slots are
    # (block, coefficient) pairs; within one coefficient the block
    # indices are unique, so the two scatter-adds cannot collide.
    signs = np.repeat([alpha if b else -alpha for b in frame_bits], REPETITION)
    slots = _PERM[:FRAME_BITS * REPETITION]
    blocks = _blocks_of(gray)
    for coef in (0, 1):
        mask = (slots % 2) == coef
        blocks[slots[mask] // 2] += signs[mask, None] * _KERNELS[coef]
    out = (blocks.reshape(BLOCKS_PER_SIDE, BLOCKS_PER_SIDE, BLOCK, BLOCK)
           .transpose(0, 2, 1, 3)
           .reshape(CANVAS_SIZE, CANVAS_SIZE))

    ys, xs = np.mgrid[0:CANVAS_SIZE, 0:CANVAS_SIZE]
    out += PILOT_AMP * np.cos(2.0 * math.pi * (PILOT_FX * xs + PILOT_FY * ys))
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "L"), alpha


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class NotDetectedError(Exception):
    """Raised when no synchronisation geometry passes the threshold."""


def _rectify(gray_img, scale, off_x, off_y):
    """Affine-rectify so canvas pixel (x, y) maps to received
    (scale*x + off_x, scale*y + off_y)."""
    return gray_img.transform(
        (CANVAS_SIZE, CANVAS_SIZE), Image.AFFINE,
        (scale, 0.0, off_x, 0.0, scale, off_y),
        resample=Image.BILINEAR, fillcolor=128)


def _frame_soft_bits(gray):
    """Soft value of every frame bit from a rectified canvas array."""
    vals = _slot_values(gray)
    take = _PERM[:FRAME_BITS * REPETITION]
    return vals[take].reshape(FRAME_BITS, REPETITION).mean(axis=1)


def _sync_score_fast(gray):
    """Sync correlation of a rectified canvas array."""
    vals = _slot_values(gray)[_SYNC_SLOTS].reshape(SYNC_BITS, REPETITION).mean(axis=1)
    denom = float(np.sqrt(np.mean(vals * vals))) + 1e-9
    return float(np.mean(vals * _SYNC_SIGNS) / denom)


def _decode_at(gray_img, scale, off_x, off_y):
    rect = _rectify(gray_img, scale, off_x, off_y)
    gray = np.asarray(rect, dtype=np.float64)
    soft = _frame_soft_bits(gray)
    sync = soft[:SYNC_BITS]
    denom = float(np.sqrt(np.mean(sync * sync))) + 1e-9
    return float(np.mean(sync * _SYNC_SIGNS) / denom), soft


def _sync_score_at(gray_img, scale, off_x, off_y):
    rect = _rectify(gray_img, scale, off_x, off_y)
    return _sync_score_fast(np.asarray(rect, dtype=np.float64))


def _frame_energy(gray):
    """Mean |coefficient| over all slots: high when the embedded pattern
    is geometrically aligned, with a clean unimodal peak at the true
    geometry (unlike the oscillating signed sync correlation)."""
    return float(np.mean(np.abs(_slot_values(gray))))


def _detect_pilot_scale(gray):
    """Estimate geometric scale from the pilot tone's FFT peak.

    Returns (scale, prominence).  The pilot sits at (PILOT_FX, PILOT_FY)
    cycles/pixel in canvas coordinates, so a peak at frequency f implies
    scale = PILOT_F / f.
    """
    h, w = gray.shape
    windowed = (gray - gray.mean()) * np.outer(np.hanning(h), np.hanning(w))
    mag = np.abs(np.fft.fft2(windowed))

    s_lo, s_hi = 0.60, 1.70
    kx0, kx1 = int(PILOT_FX / s_hi * w), math.ceil(PILOT_FX / s_lo * w)
    ky0, ky1 = int(PILOT_FY / s_hi * h), math.ceil(PILOT_FY / s_lo * h)
    region = mag[ky0:ky1 + 1, kx0:kx1 + 1]
    if region.size == 0:
        return None, 0.0
    iy, ix = np.unravel_index(int(np.argmax(region)), region.shape)
    ky, kx = iy + ky0, ix + kx0

    def _subbin(minus, center, plus):
        denom = minus - 2.0 * center + plus
        return 0.0 if denom == 0 else 0.5 * (minus - plus) / denom

    dx = _subbin(mag[ky, kx - 1], mag[ky, kx], mag[ky, kx + 1])
    dy = _subbin(mag[ky - 1, kx], mag[ky, kx], mag[ky + 1, kx])
    fx = (kx + dx) / w
    fy = (ky + dy) / h
    if fx <= 0 or fy <= 0:
        return None, 0.0
    scale = 0.5 * (PILOT_FX / fx + PILOT_FY / fy)
    prominence = float(mag[ky, kx] / (np.median(region) + 1e-9))
    return scale, prominence


def _search_geometry(gray_img):
    """Pilot-guided search of (scale, off_x, off_y); returns best tuple.

    Scale comes from the pilot tone (plus an energy-based fine sweep);
    translation is refined by maximising the frame energy, whose
    landscape is unimodal at the true geometry.  Energy is evaluated on
    the centre 512x512 canvas region only (always preserved by centre
    crops), which keeps every evaluation cheap.  The returned tuple is
    (sync_score, scale, off_x, off_y) where sync_score is the signed
    sync-word correlation used only as a final verification gate.
    """
    w, h = gray_img.size
    gray = np.asarray(gray_img, dtype=np.float64)
    scale_est, prominence = _detect_pilot_scale(gray)

    center = (CANVAS_SIZE - _SEARCH_REGION) // 2
    region_blocks = _SEARCH_REGION // BLOCK

    def energy_of(arr):
        blocks = (arr.reshape(region_blocks, BLOCK, region_blocks, BLOCK)
                  .transpose(0, 2, 1, 3)
                  .reshape(region_blocks * region_blocks, BLOCK * BLOCK))
        e0 = np.abs(blocks @ _KERNELS[0]).mean()
        e1 = np.abs(blocks @ _KERNELS[1]).mean()
        return float(e0 + e1)

    def energy_at(s, ox, oy):
        rect = gray_img.transform(
            (_SEARCH_REGION, _SEARCH_REGION), Image.AFFINE,
            (s, 0.0, ox + s * center, 0.0, s, oy + s * center),
            resample=Image.BILINEAR, fillcolor=128)
        return energy_of(np.asarray(rect, dtype=np.float64))

    def centered_offsets(s):
        return (w / 2.0 - s * CANVAS_SIZE / 2.0,
                h / 2.0 - s * CANVAS_SIZE / 2.0)

    # ---- scale sweep around the pilot estimate (fallback: full range) --
    if scale_est is not None and prominence >= PILOT_MIN_PROMINENCE:
        scales = np.arange(scale_est - 0.003, scale_est + 0.0031, 0.0005)
    else:
        scales = np.arange(0.66, 1.581, 0.005)
    best = None  # (energy, scale, ox, oy)
    for s in scales:
        ox, oy = centered_offsets(s)
        e = energy_at(s, ox, oy)
        if best is None or e > best[0]:
            best = (e, float(s), ox, oy)
    e0, s1, ox1, oy1 = best

    # ---- translation polish: grid search on shifted windows of one big
    # rectified region, then parabolic sub-pixel interpolation.  Shifting
    # the output window by 1px shifts the effective input offset by s1
    # px, so the big region is anchored at ox1 + s1*(center - margin) and
    # a window shift of (dx, dy) evaluates offset (ox1 + s1*dx, ...). ---
    margin = 4
    big = gray_img.transform(
        (_SEARCH_REGION + 2 * margin, _SEARCH_REGION + 2 * margin), Image.AFFINE,
        (s1, 0.0, ox1 + s1 * (center - margin), 0.0, s1, oy1 + s1 * (center - margin)),
        resample=Image.BILINEAR, fillcolor=128)
    arr = np.asarray(big, dtype=np.float64)

    def window_energy(dx, dy):
        return energy_of(arr[margin + dy:margin + dy + _SEARCH_REGION,
                             margin + dx:margin + dx + _SEARCH_REGION])

    best_dx, best_dy, best_e = 0, 0, window_energy(0, 0)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            e = window_energy(dx, dy)
            if e > best_e:
                best_dx, best_dy, best_e = dx, dy, e

    def parabolic(em, e0, ep):
        denom = em - 2.0 * e0 + ep
        return 0.0 if denom == 0 else 0.5 * (em - ep) / denom

    fx = best_dx + 0.5 * parabolic(window_energy(best_dx - 1, best_dy), best_e,
                                   window_energy(best_dx + 1, best_dy))
    fy = best_dy + 0.5 * parabolic(window_energy(best_dx, best_dy - 1), best_e,
                                   window_energy(best_dx, best_dy + 1))
    ox2, oy2 = ox1 + s1 * fx, oy1 + s1 * fy

    # ---- scale micro-polish (offsets track the canvas centre) ---------
    best_s, best_se = s1, energy_at(s1, ox2, oy2)
    for ds in (-0.001, -0.0005, 0.0005, 0.001):
        s2 = s1 + ds
        e = energy_at(s2, ox2 - CANVAS_SIZE / 2 * ds, oy2 - CANVAS_SIZE / 2 * ds)
        if e > best_se:
            best_s, best_se = s2, e
    if best_s != s1:
        ds = best_s - s1
        ox2, oy2 = ox2 - CANVAS_SIZE / 2 * ds, oy2 - CANVAS_SIZE / 2 * ds

    score = _sync_score_at(gray_img, best_s, ox2, oy2)
    return score, best_s, ox2, oy2


def extract(image_gray, min_sync_score=0.30):
    """Locate and decode the watermark in a transformed image.

    Returns a dict with payload bytes, confidence and geometry.  Raises
    ``NotDetectedError`` if no plausible geometry is found and
    ``PayloadError``/``RSDecodeError`` if the payload cannot be recovered.
    """
    gray_img = image_gray.convert("L")
    score, scale, ox, oy = _search_geometry(gray_img)
    if score < min_sync_score:
        raise NotDetectedError(f"sync not found (best score {score:.3f})")

    _, soft = _decode_at(gray_img, scale, ox, oy)
    bits = [1 if v > 0 else 0 for v in soft[SYNC_BITS:]]
    codeword = _bits_to_bytes(bits)
    payload, corrected = rs_decode(list(codeword), RS_NSYM)

    # confidence: agreement between received hard bits and the clean
    # re-encoded codeword (1.0 = every bit already correct).
    clean = rs_encode(payload, RS_NSYM)
    clean_bits = _bytes_to_bits(clean)
    agree = sum(1 for a, b in zip(bits, clean_bits) if a == b)
    confidence = agree / len(clean_bits)

    return {
        "payload": bytes(payload),
        "confidence": confidence,
        "rs_corrected": corrected,
        "sync_score": score,
        "scale": scale,
        "offset_x": ox,
        "offset_y": oy,
    }
