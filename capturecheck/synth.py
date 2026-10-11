"""Synthetic contributor batch with planted problems and hard negatives.

Everything is invented. A "recording" is what a phone on a head strap would hand to the
pipeline after decoding: low-res grayscale thumbnails at 4 fps, a 50 Hz gyro, a per-frame
hand-keypoint confidence and the capture app's camera/IMU calibration.
"""
import numpy as np

from .media import FPS, resize

H, W = 24, 32          # thumbnail size
PH, PW = 32, 128       # environment panorama (wraps horizontally)
GYRO_HZ = 50
PX2RAD = 1.2 / W       # ~69 degree horizontal field of view
RATES = {"home": 5.0, "work": 6.0}
BOOSTED = 8.0

MODELS = {  # nominal intrinsics for three phone models (full-res pixels)
    "phone-A": (1450.0, 1450.0, 960.0, 540.0),
    "phone-B": (1510.0, 1510.0, 960.0, 540.0),
    "phone-C": (1395.0, 1395.0, 960.0, 540.0),
}
CALIB_NOISE = np.array([7e-4] * 3 + [2e-3] * 3 + [0.9, 0.9, 0.3, 0.3])  # per-recording estimate noise


def _smooth_noise(rng, n, lo, hi, k=3):
    t = np.arange(n) / FPS
    s = sum(np.sin(2 * np.pi * rng.uniform(lo, hi) * t + rng.uniform(0, 2 * np.pi)) * rng.uniform(0.5, 1)
            for _ in range(k))
    return s / k * 1.6


def make_panorama(rng):
    low = rng.random((PH // 4, PW // 4))
    pano = np.kron(low, np.ones((4, 4))) * 0.5
    for _ in range(14):  # cabinets, appliances, shelves
        y, x = rng.integers(0, PH - 4), rng.integers(0, PW)
        h, w = rng.integers(3, 14), rng.integers(4, 22)
        cols = np.arange(x, x + w) % PW
        pano[y:y + h][:, cols] = rng.uniform(0.1, 0.95)
    k = np.ones(3) / 3  # light blur, wrapping horizontally
    pano = np.stack([np.convolve(np.r_[r[-1:], r, r[:1]], k, "valid") for r in pano])
    return np.clip(pano, 0, 1)


class Env:
    def __init__(self, rng, name, kind, pano=None):
        self.name, self.kind = name, kind
        self.pano = make_panorama(rng) if pano is None else pano
        self.spot = rng.uniform(0, PW)
        self.roam = 14.0 if kind == "work" else 10.0


class Device:
    def __init__(self, rng, name):
        self.name = name
        self.model = rng.choice(sorted(MODELS))
        fx, fy, cx, cy = MODELS[self.model]
        self.calib = np.r_[rng.normal(0, 0.02, 3), rng.normal(0, 0.05, 3),
                           fx * (1 + rng.normal(0, 0.004)), fy * (1 + rng.normal(0, 0.004)),
                           cx + rng.normal(0, 2), cy + rng.normal(0, 2)]


def _render(env, yaw, pitch, hands, gain, rng):
    T = len(yaw)
    cols = yaw[:, None] + np.arange(W)[None, :]
    c0 = np.floor(cols).astype(int); fr = cols - c0
    rows = np.clip(np.round(pitch).astype(int), 0, PH - H)[:, None] + np.arange(H)[None, :]
    p = env.pano
    img = (p[rows[:, :, None], (c0 % PW)[:, None, :]] * (1 - fr)[:, None, :]
           + p[rows[:, :, None], ((c0 + 1) % PW)[:, None, :]] * fr[:, None, :])
    yy, xx = np.mgrid[0:H, 0:W]
    for hx, hy in hands:
        img = img + 0.7 * np.exp(-((xx[None] - hx[:, None, None]) ** 2 + (yy[None] - hy[:, None, None]) ** 2) / (2 * 2.2 ** 2))
    img = np.clip(img, 0, 1) * gain + rng.normal(0, 0.015, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def _gyro(yaw, pitch, rng, calib, shaky=False):
    n = int(len(yaw) / FPS * GYRO_HZ)
    tf = np.arange(len(yaw)) / FPS
    tg = np.arange(n) / GYRO_HZ
    yaw_rate = np.gradient(np.interp(tg, tf, yaw)) * GYRO_HZ * PX2RAD
    pitch_rate = np.gradient(np.interp(tg, tf, pitch)) * GYRO_HZ * PX2RAD
    g = np.stack([pitch_rate, yaw_rate, np.zeros(n)], 1)
    g += 0.03 * rng.standard_normal((n, 3)) + 0.04 * np.sin(2 * np.pi * 1.8 * tg)[:, None]
    if shaky:
        f = rng.uniform(5, 8)
        g += 0.5 * np.sin(2 * np.pi * f * tg)[:, None] * rng.uniform(0.6, 1, 3) + 0.25 * rng.standard_normal((n, 3))
    return (g + calib[:3]).astype(np.float32)


def record(rng, rid, env, device, account, site, minutes, *, static=False, shaky=False, dark=False,
           nohands=False, idle=None, boosted=False, uploaded_at=0):
    T = int(minutes * 60 * FPS)
    yaw = env.spot + (1.0 if static else env.roam) * _smooth_noise(rng, T, 0.01, 0.08)
    if nohands:
        yaw = env.spot + 30 * _smooth_noise(rng, T, 0.01, 0.05)
    pitch = 4 + 1.5 * _smooth_noise(rng, T, 0.02, 0.1)
    hx = 13 + 9 * _smooth_noise(rng, T, 0.08, 0.5)
    hy = 15 + 5 * _smooth_noise(rng, T, 0.08, 0.5)
    hx2 = hx + 9 + 3 * _smooth_noise(rng, T, 0.1, 0.6)
    hy2 = hy + 1 + 3 * _smooth_noise(rng, T, 0.1, 0.6)
    off = np.zeros(T, bool)
    n_p = int(rng.integers(1, 4))                             # natural short pauses (paid), apart
    for k, a in enumerate(np.sort(rng.choice(np.arange(0, T - 12 * FPS, 15 * FPS), n_p, replace=False))):
        off[a:a + int(rng.uniform(3, 10) * FPS)] = True
    if nohands:
        off[:] = True
        for _ in range(3):
            a = rng.integers(0, T - 20 * FPS); off[a:a + int(rng.uniform(5, 12) * FPS)] = False
    idle_mask = np.zeros(T, bool)
    if idle:                                                  # padding: phone still, nobody working
        at, secs = idle
        n = int(secs * FPS); a = int(at * T)
        yaw = np.r_[yaw[:a], np.full(n, yaw[a]) + rng.normal(0, 0.05, n), yaw[a:]]
        pitch = np.r_[pitch[:a], np.full(n, pitch[a]), pitch[a:]]
        hx, hy, hx2, hy2 = (np.r_[v[:a], np.full(n, v[a]), v[a:]] for v in (hx, hy, hx2, hy2))
        off = np.r_[off[:a], np.ones(n, bool), off[a:]]
        idle_mask = np.r_[idle_mask[:a], np.ones(n, bool), idle_mask[a:]]
        T += n
    jitter_y = jitter_p = 0
    if shaky:
        jitter_y = rng.normal(0, 2.5, T); jitter_p = rng.normal(0, 1.5, T)
    hy_v = np.where(off, 60.0, hy); hy2_v = np.where(off, 60.0, hy2)
    gain = 0.28 if dark else rng.uniform(0.75, 1.1)
    frames = _render(env, yaw + jitter_y, pitch + jitter_p, [(hx, hy_v), (hx2, hy2_v)], gain, rng)
    conf = np.where(off, rng.uniform(0, 0.15, T), rng.uniform(0.82, 0.97, T)).astype(np.float32)
    calib = device.calib + rng.normal(0, 1, 10) * CALIB_NOISE
    rate = BOOSTED if boosted else RATES[env.kind]
    return {"id": rid, "account": account, "site": site, "kind": env.kind, "rate": rate,
            "uploaded_at": uploaded_at, "frames": frames, "hand_conf": conf,
            "gyro": _gyro(yaw + jitter_y, pitch + jitter_p, rng, device.calib, shaky),
            "calib": calib, "duration_s": T / FPS,
            "_yaw": yaw, "_pitch": pitch, "_idle": idle_mask}


TRANSFORMS = ["reencode", "crop", "mirror", "speed", "trim", "combo"]


def make_copy(rng, src, rid, account, site, device, how, uploaded_at):
    f, conf = src["frames"].astype(np.float64), src["hand_conf"]
    yaw, pitch = src["_yaw"], src["_pitch"]
    idx = np.arange(len(f))
    steps = {"reencode": ["reencode"], "crop": ["crop", "reencode"], "mirror": ["mirror", "reencode"],
             "speed": ["speed", "reencode"], "trim": ["trim", "reencode"],
             "combo": ["mirror", "crop", "speed", "trim", "reencode"]}[how]
    info = {}
    if "trim" in steps:
        a = int(len(idx) * rng.uniform(0.1, 0.3)); b = int(len(idx) * rng.uniform(0.75, 0.95))
        idx = idx[a:b]
    if "speed" in steps:
        r = float(rng.choice([rng.uniform(1.08, 1.3), rng.uniform(0.78, 0.92)]))
        info["speed"] = round(r, 3)
        pos = np.arange(idx[0], idx[-1], r)
        idx = np.round(pos).astype(int)
    f, conf, yaw, pitch = f[idx], conf[idx], yaw[idx], pitch[idx]
    if "crop" in steps:
        c = rng.uniform(0.84, 0.93); info["crop"] = round(c, 2)
        dh, dw = int(H * (1 - c) / 2), int(W * (1 - c) / 2)
        f = resize(f[:, dh:H - dh, dw:W - dw], H, W)
    if "mirror" in steps:
        f = f[:, :, ::-1]; yaw = -yaw
    if "reencode" in steps:
        f = np.round((f * rng.uniform(0.85, 1.15) + rng.normal(0, 0.02, f.shape)) * 24) / 24
    f = np.clip(f, 0, 1).astype(np.float32)
    rec = {"id": rid, "account": account, "site": site, "kind": src["kind"], "rate": src["rate"],
           "uploaded_at": uploaded_at, "frames": f, "hand_conf": conf,
           "gyro": _gyro(yaw, pitch, rng, device.calib), "calib": device.calib + rng.normal(0, 1, 10) * CALIB_NOISE,
           "duration_s": len(f) / FPS, "_yaw": yaw, "_pitch": pitch, "_idle": src["_idle"][idx]}
    return rec, info


def make_batch(seed):
    """Returns (recordings, truth). The layout is fixed; everything inside it is random per seed."""
    rng = np.random.default_rng(seed)
    envs = {n: Env(rng, n, "home") for n in ["home-1", "home-2", "home-3", "home-4", "home-5"]}
    look = Env(rng, "home-6", "home", pano=np.clip(0.55 * envs["home-5"].pano + 0.45 * make_panorama(rng), 0, 1))
    look.spot = envs["home-5"].spot          # look-alike room, same angle: hard negative
    envs["home-6"] = look
    for n in ["kitchen", "warehouse", "auto-shop"]:
        envs[n] = Env(rng, n, "work")
    devices = {f"dev-{i}": Device(rng, f"dev-{i}") for i in range(1, 13)}

    # person -> (env, device, [(account, site)])
    people = {
        "p1": ("home-1", "dev-1", [("acct-101", "H-101")]),
        "p2": ("home-2", "dev-2", [("acct-102", "H-102")]),          # household with p3,
        "p3": ("home-2", "dev-3", [("acct-103", "H-103")]),          # different phones: fine
        "p4": ("home-3", "dev-4", [("acct-104", "H-104"), ("acct-188", "H-188")]),  # 2 accounts, 1 phone
        "p5": ("home-4", "dev-5", [("acct-105", "H-105")]),          # static camera: hard negative
        "p6": ("home-5", "dev-6", [("acct-106", "H-106")]),
        "p7": ("home-6", "dev-7", [("acct-107", "H-107")]),          # look-alike of home-5
        "p8": ("kitchen", "dev-8", [("acct-108", "S-301")]),         # one kitchen registered
        "p9": ("kitchen", "dev-9", [("acct-109", "S-302")]),         # as three sites
        "p10": ("kitchen", "dev-10", [("acct-110", "S-303")]),
        "p11": ("warehouse", "dev-11", [("acct-111", "S-401")]),
        "p12": ("auto-shop", "dev-12", [("acct-112", "S-402")]),
    }
    recs, truth = [], {"env": {}, "person": {}, "dups": [], "quality": {}, "idle": {},
                       "multi_account": [{"acct-104", "acct-188"}],
                       "site_alias": [{"S-301", "S-302", "S-303"}, {"H-102", "H-103"}, {"H-104", "H-188"}]}
    for p, (_, _, accts) in people.items():
        for a, _ in accts:
            truth["person"][a] = p

    plan = []
    for p, (env, dev, accts) in people.items():
        n = int(rng.integers(3, 5))
        for i in range(n):
            a, s = accts[i % len(accts)]
            plan.append((p, env, dev, a, s))
    order = rng.permutation(len(plan))
    plan = [plan[i] for i in order]
    clean_idx = [i for i, x in enumerate(plan) if x[0] not in ("p5",)]
    special = rng.choice(clean_idx, 5, replace=False)
    flags = dict(zip(special, ["shaky", "dark", "nohands", "idle", "idle"]))

    t = 0
    for i, (p, env, dev, a, s) in enumerate(plan):
        t += int(rng.integers(5, 90))
        rid = f"rec-{i + 1:03d}"
        kw = {}
        fl = flags.get(i)
        if fl in ("shaky", "dark", "nohands"):
            kw[fl] = True
            truth["quality"][rid] = {"shaky": "STEADY", "dark": "LIGHT", "nohands": "HANDS"}[fl]
        if fl == "idle":
            secs = float(rng.uniform(60, 150)); kw["idle"] = (rng.uniform(0.3, 0.8), secs)
            truth["idle"][rid] = secs
        rec = record(rng, rid, envs[env], devices[dev], a, s, rng.uniform(2, 4), static=(p == "p5"),
                     boosted=bool(rng.random() < 0.2), uploaded_at=t, **kw)
        recs.append(rec); truth["env"][rid] = env

    # re-uploads, later in the batch, by the same person or by someone else
    sources = [r for r in recs if r["id"] not in truth["quality"] and r["id"] not in truth["idle"]]
    for j, how in enumerate(TRANSFORMS):
        src = sources[int(rng.integers(len(sources)))]
        p = rng.choice(sorted(people))
        env, dev, accts = people[p]
        a, s = accts[0]
        t += int(rng.integers(30, 300))
        rid = f"rec-{len(recs) + 1:03d}"
        rec, info = make_copy(rng, src, rid, a, s, devices[dev], how, t)
        recs.append(rec); truth["env"][rid] = truth["env"][src["id"]]
        truth["dups"].append({"orig": src["id"], "copy": rid, "how": how, **info,
                              "copy_s": rec["duration_s"]})
    return recs, truth
