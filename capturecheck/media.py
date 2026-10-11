"""Frame-level primitives: resize, perceptual hash, popcount, motion profile."""
import numpy as np

FPS = 4  # thumbnail frames per second used for checks


def resize(frames, h, w):
    """Bilinear resize of a (T, H, W) stack."""
    T, H, W = frames.shape
    ys = np.linspace(0, H - 1, h)
    xs = np.linspace(0, W - 1, w)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, H - 1); fy = (ys - y0)[None, :, None]
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, W - 1); fx = (xs - x0)[None, None, :]
    top = frames[:, y0][:, :, x0] * (1 - fx) + frames[:, y0][:, :, x1] * fx
    bot = frames[:, y1][:, :, x0] * (1 - fx) + frames[:, y1][:, :, x1] * fx
    return top * (1 - fy) + bot * fy


_BITS = (np.uint64(1) << np.arange(64, dtype=np.uint64))


def dhash(frames):
    """64-bit difference hash per frame: 8x9 area-averaged grid, compare horizontal neighbours."""
    T, H, W = frames.shape
    rows = np.array_split(np.arange(H), 8)
    cols = np.array_split(np.arange(W), 9)
    small = np.stack([np.stack([frames[:, r][:, :, c].mean(axis=(1, 2)) for c in cols], -1)
                      for r in rows], 1)                       # (T, 8, 9)
    bits = (small[:, :, 1:] > small[:, :, :-1]).reshape(T, 64)
    return (bits.astype(np.uint64) * _BITS).sum(axis=1, dtype=np.uint64)


def popcount(x):
    """Bit count of a uint64 array."""
    x = x.astype(np.uint64)
    m1, m2, m4 = np.uint64(0x5555555555555555), np.uint64(0x3333333333333333), np.uint64(0x0F0F0F0F0F0F0F0F)
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def hamming_matrix(a, b):
    return popcount(a[:, None] ^ b[None, :])


def informative(frames, min_std=0.03):
    """Frames with enough texture to hash (dark or flat frames give degenerate hashes)."""
    return frames.reshape(len(frames), -1).std(axis=1) >= min_std


def motion_profile(frames):
    """Mean absolute change between consecutive frames on a 6x8 block grid, contrast-normalised.

    Block means average away sensor noise, so the profile follows camera and hand motion.
    It survives mirroring, mild crops, re-encoding and brightness changes; a speed change only
    rescales time, which the aligner searches over.
    """
    T, H, W = frames.shape
    s = frames[:, :H - H % 6, :W - W % 8].reshape(T, 6, H // 6, 8, W // 8).mean(axis=(2, 4))
    s = s / (s.mean(axis=(1, 2))[:, None, None] + 1e-6)
    d = np.abs(np.diff(s, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[d[0] if len(d) else 0.0], d])


def fmt_t(sec):
    sec = int(round(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"
