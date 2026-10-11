"""Cross-recording checks: candidate pairs from an LSH index, shared-place score, re-upload test.

Two recordings in the same kitchen share background frames, so frame similarity alone cannot
tell "same place" from "same footage". A re-upload also repeats the *motion*: the same hands
and head movement in the same order. So a pair is a duplicate only when its motion profiles line
up along one straight time mapping (any speed, any offset), checked chunk by chunk.
"""
from collections import defaultdict

import numpy as np

from .media import FPS, dhash, hamming_matrix, informative, motion_profile, popcount, resize

TABLES = 10               # bit-sampling LSH tables
KEY_BITS = 20             # bits sampled per table
BUCKET_CAP = 2000         # skip buckets this full (flat, generic views)
MIN_HITS = 5              # query frames that must share a band with the other recording
NEAR = 8                  # Hamming distance for "same view"
PLACE_NEAR = 7            # stricter distance for "same place"
PLACE_NCC = 0.9           # ...and the 12x16 thumbnails must agree this well
PLACE_SHARE = 0.025       # share of query frames with a verified same-place view
RATES = np.geomspace(0.75, 1 / 0.75, 11)
CHUNK = 40                # 10 s of motion profile
CHUNK_NCC = 0.75
MIN_OVERLAP_S = 30
FULL_NCC = 0.6


def _thumbs(f):
    t = resize(f.astype(np.float64), 12, 16).reshape(len(f), -1)
    t = t - t.mean(axis=1, keepdims=True)
    return (t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


class Signatures:
    def __init__(self, rec):
        f = rec["frames"]
        keep = informative(f)
        self.id = rec["id"]
        self.all = dhash(f)[keep]                       # every frame, as indexed
        self.all_m = dhash(f[:, :, ::-1])[keep]          # mirrored view
        q = np.zeros(len(f), bool); q[::FPS] = True     # query at 1 fps
        self.q = dhash(f[q & keep])
        self.q_thumb = _thumbs(f[q & keep])
        self.all_thumb = _thumbs(f[keep])
        self.profile = motion_profile(f)


_TABLES = np.random.default_rng(7).permuted(np.tile(np.arange(64), (TABLES, 1)), axis=1)[:, :KEY_BITS]


def _bits(h):
    return ((h[:, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)).astype(np.int64)


def _keys(h):
    """Bit-sampling LSH: one KEY_BITS-bit key per table per hash."""
    b = _bits(h)
    w = 1 << np.arange(KEY_BITS, dtype=np.int64)
    return [b[:, cols] @ w for cols in _TABLES]


def candidates(sigs, stats=None):
    """Pairs of recordings with at least MIN_HITS query frames that have a verified near view.

    Every hash goes into TABLES bit-sampled tables; a query only looks at its buckets and then
    checks the real Hamming distance, so the work grows with matches, not with recordings squared.
    """
    owner = np.concatenate([np.full(len(s.all) * 2, i) for i, s in enumerate(sigs)])
    hashes = np.concatenate([np.r_[s.all, s.all_m] for s in sigs]).astype(np.uint64)
    tables = []
    for keys in _keys(hashes):
        order = np.argsort(keys, kind="stable")
        k = keys[order]
        uniq, first, counts = np.unique(k, return_index=True, return_counts=True)
        tables.append((dict(zip(uniq.tolist(), zip(first.tolist(), counts.tolist()))), order))
    hits = defaultdict(int)
    checked = 0
    for i, s in enumerate(sigs):
        if not len(s.q):
            continue
        qkeys = _keys(s.q)
        for f in range(len(s.q)):
            cand = []
            for t, (lookup, order) in enumerate(tables):
                e = lookup.get(int(qkeys[t][f]))
                if e and e[1] <= BUCKET_CAP:
                    cand.append(order[e[0]:e[0] + e[1]])
            if not cand:
                continue
            cand = np.unique(np.concatenate(cand))
            cand = cand[owner[cand] != i]
            checked += len(cand)
            near = cand[popcount(hashes[cand] ^ s.q[f]) <= NEAR]
            for o in np.unique(owner[near]):
                hits[(min(i, o), max(i, o))] += 1
    if stats is not None:
        stats["frame_checks"] = checked
    return sorted((sigs[a].id, sigs[b].id) for (a, b), n in hits.items() if n >= MIN_HITS)


def place_share(a, b):
    """Share of query frames of one recording with a verified view of the other, symmetric max.

    A view is verified when the hash is close (PLACE_NEAR) and the 12x16 thumbnails correlate
    (PLACE_NCC). The hash alone also matches look-alike rooms; the thumbnail check does not.
    """
    def one(x, y):
        if not len(x.q) or not len(y.all):
            return 0.0
        close = hamming_matrix(x.q, y.all) <= PLACE_NEAR
        if not close.any():
            return 0.0
        ncc = x.q_thumb @ y.all_thumb.T
        ncc[~close] = -1
        return float((ncc.max(axis=1) >= PLACE_NCC).mean())
    return max(one(a, b), one(b, a))


def _smooth(p, k=5):
    """Box-smooth the motion profile: a speed change repeats or skips frames, which puts zeros or
    double steps into the raw frame differences."""
    return np.convolve(np.r_[np.full(k // 2, p[0]), p, np.full(k // 2, p[-1])], np.ones(k) / k, "valid")


def _z(x):
    x = x - x.mean(axis=-1, keepdims=True)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _theil_sen(x, y):
    i, j = np.triu_indices(len(x), 1)
    ok = x[j] != x[i]
    slope = np.median((y[j] - y[i])[ok] / (x[j] - x[i])[ok])
    return slope, np.median(y - slope * x)


def align(orig, copy):
    """Find a straight time mapping orig_t = rate * copy_t + offset where motion repeats.

    Returns None, or {rate, offset_s, overlap_s, chunks, inliers, ncc}.
    """
    pa, pb = _smooth(orig.profile), _smooth(copy.profile)
    if len(pa) < CHUNK * 2 or len(pb) < CHUNK * 2:
        return None
    starts = np.arange(0, len(pb) - CHUNK + 1, CHUNK // 2)
    chunks = _z(np.stack([pb[s:s + CHUNK] for s in starts]))
    best = np.full(len(starts), -1.0); pos = np.zeros(len(starts))
    for r in RATES:
        k = np.arange(0, (len(pa) - 1) / r)
        ar = np.interp(k * r, np.arange(len(pa)), pa)
        if len(ar) < CHUNK:
            continue
        win = _z(np.lib.stride_tricks.sliding_window_view(ar, CHUNK))
        ncc = chunks @ win.T
        j = ncc.argmax(axis=1); v = ncc[np.arange(len(starts)), j]
        better = v > best
        best[better] = v[better]
        pos[better] = (j[better] + CHUNK / 2) * r          # chunk centre, in orig samples
    good = best >= CHUNK_NCC
    if good.sum() < 3:
        return None
    xb = starts[good] + CHUNK / 2; ya = pos[good]
    rate, off = _theil_sen(xb.astype(float), ya)
    if not (RATES[0] * 0.95 <= rate <= RATES[-1] * 1.05):
        return None
    inl = np.abs(ya - (rate * xb + off)) <= 2 * max(rate, 1)
    if inl.sum() < 3:
        return None
    lo, hi = xb[inl].min() - CHUNK / 2, xb[inl].max() + CHUNK / 2
    covered = (starts + CHUNK / 2 >= lo) & (starts + CHUNK / 2 <= hi)
    if inl.sum() < 0.5 * covered.sum():
        return None
    j = np.arange(int(lo), int(hi))
    src = rate * j + off
    ok = (src >= 0) & (src <= len(pa) - 1)
    if ok.sum() < MIN_OVERLAP_S * FPS:
        return None
    a = np.interp(src[ok], np.arange(len(pa)), pa)
    ncc = float(_z(a) @ _z(pb[j[ok]]))
    if ncc < FULL_NCC:
        return None
    return {"rate": float(rate), "offset_s": float(off / FPS), "overlap_s": float(ok.sum() / FPS),
            "chunks": int(covered.sum()), "inliers": int(inl.sum()), "ncc": ncc}
