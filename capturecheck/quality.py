"""Per-recording checks for Hub's published passing bar: "steady first-person view, hands visible,
good light" (https://ai.hub.xyz/facts), plus payable time.

Every reject carries a reason with timestamps, because Hub promises "if rejected, the reason".
"""
import numpy as np

from .media import FPS, fmt_t
from .synth import GYRO_HZ

SHAKE_RMS = 0.25        # rad/s, high-passed gyro RMS in a 10 s window
SHAKE_SHARE = 0.25      # reject when more than this share of windows is shaky
MIN_LUMA = 0.19         # median frame brightness (0..1)
HAND_CONF = 0.5
MIN_HANDS = 0.5         # share of paid time with hands visible
IDLE_MIN_S = 15         # hands-off stretch that is not paid
STILL = 0.03            # rad/s, median 1 s-smoothed camera rotation of an idle stretch


def _runs(mask):
    """(start, end) index pairs of True runs."""
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def check(rec):
    out = {"id": rec["id"], "reasons": [], "idle": [], "status": "PASS"}
    dur = rec["duration_s"]

    g = rec["gyro"].astype(np.float64)
    k = np.ones(25) / 25
    hp = g - np.stack([np.convolve(g[:, i], k, "same") for i in range(3)], 1)
    w = 10 * GYRO_HZ
    rms = np.array([np.sqrt((hp[i:i + w] ** 2).sum(1).mean()) for i in range(0, len(hp) - w + 1, w)])
    shaky = rms > SHAKE_RMS
    out["shake_rms"] = float(np.median(rms)) if len(rms) else 0.0
    if len(rms) and shaky.mean() > SHAKE_SHARE:
        worst = int(np.argmax(rms))
        out["reasons"].append(("STEADY", f"camera shake in {shaky.sum()} of {len(rms)} 10-second windows; "
                               f"worst {fmt_t(worst * 10)}-{fmt_t(worst * 10 + 10)} at {rms[worst]:.2f} rad/s "
                               f"(limit {SHAKE_RMS})"))

    luma = float(np.median(rec["frames"].mean(axis=(1, 2))))
    out["luma"] = luma
    if luma < MIN_LUMA:
        out["reasons"].append(("LIGHT", f"too dark: median brightness {luma:.2f} (needs {MIN_LUMA})"))

    # camera rotation per frame: 1 s moving average of the gyro, bias removed
    k1 = np.ones(GYRO_HZ) / GYRO_HZ
    lp = np.stack([np.convolve(g[:, i] - np.median(g[:, i]), k1, "same") for i in range(3)], 1)
    speed = np.linalg.norm(lp, axis=1)
    speed_f = np.interp(np.arange(len(rec["hand_conf"])) / FPS, np.arange(len(speed)) / GYRO_HZ, speed)

    off = rec["hand_conf"] < HAND_CONF
    idle_mask = np.zeros(len(off), bool)
    for a, b in _runs(off):  # nobody working: hands gone AND the phone is still
        if (b - a) / FPS >= IDLE_MIN_S and np.median(speed_f[a:b]) < STILL:
            idle_mask[a:b] = True
            out["idle"].append((a / FPS, b / FPS))
    idle_s = idle_mask.sum() / FPS
    active = ~idle_mask
    hands_share = float((~off[active]).mean()) if active.any() else 0.0
    out["hands_share"] = hands_share
    if hands_share < MIN_HANDS:
        a, b = max(_runs(off & active), key=lambda r: r[1] - r[0])
        out["reasons"].append(("HANDS", f"hands visible in {hands_share:.0%} of working time (needs "
                               f"{MIN_HANDS:.0%}); longest gap {fmt_t(a / FPS)}-{fmt_t(b / FPS)}"))
    out["idle_s"] = float(idle_s)
    out["payable_s"] = 0.0 if out["reasons"] else float(dur - idle_s)
    if out["reasons"]:
        out["status"] = "REJECT"
    elif idle_s:
        out["status"] = "PARTIAL"
        out["note"] = "unpaid idle (no hands): " + ", ".join(f"{fmt_t(a)}-{fmt_t(b)}" for a, b in out["idle"])
    return out
