"""Places and accounts.

- Places: recordings whose backgrounds match are one physical place, whatever site ID was claimed.
  Diversity caps are counted on places.
- Accounts: the capture app stores a camera/IMU calibration with every recording (VIO needs it).
  The same phone gives the same calibration within its repeatability, so one phone on two
  accounts is visible. This is a device fingerprint: it stays inside fraud review, is never
  shipped in a dataset, and only ever triggers a person's review, never an automatic ban.
"""
from collections import defaultdict

import numpy as np

# Repeatability of the stored calibration between two recordings on one phone
# (gyro bias rad/s x3, accel bias m/s^2 x3, fx, fy, cx, cy in px).
CALIB_REPEAT = np.array([7e-4] * 3 + [2e-3] * 3 + [0.9, 0.9, 0.3, 0.3])
SAME_DEVICE = 3.0      # RMS distance in units of repeatability


class UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)

    def groups(self):
        g = defaultdict(list)
        for i in self.p:
            g[self.find(i)].append(i)
        return sorted((sorted(v) for v in g.values()), key=lambda v: v[0])


def device_distance(ca, cb):
    return float(np.sqrt(np.mean(((ca - cb) / (CALIB_REPEAT * np.sqrt(2))) ** 2)))


def places(recs, place_edges, threshold):
    """Average-linkage clustering on the verified view share.

    place_edges: [(a, b, share)] between recordings that are not re-uploads; missing pairs are 0.
    Two groups merge only if their *average* cross share clears the threshold, so a single lucky
    match between look-alike rooms cannot chain them together (single linkage would).
    """
    by_id = {r["id"]: r for r in recs}
    share = {}
    for a, b, s in place_edges:
        share[(a, b)] = share[(b, a)] = s
    groups = [[r["id"]] for r in recs]
    while True:
        best, pair = threshold, None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                tot = sum(share.get((x, y), 0.0) for x in groups[i] for y in groups[j])
                avg = tot / (len(groups[i]) * len(groups[j]))
                if avg >= best:
                    best, pair = avg, (i, j)
        if pair is None:
            break
        i, j = pair
        groups[i] = sorted(groups[i] + groups[j])
        del groups[j]
    out = []
    for n, members in enumerate(sorted(groups, key=lambda g: g[0]), 1):
        out.append({"place": f"place-{n:02d}", "recordings": members,
                    "sites": sorted({by_id[m]["site"] for m in members}),
                    "accounts": sorted({by_id[m]["account"] for m in members})})
    return out


def site_aliases(place_list, place_edges, recs):
    """Places that were registered under more than one site ID, with the strongest evidence."""
    site = {r["id"]: r["site"] for r in recs}
    out = []
    for p in place_list:
        if len(p["sites"]) < 2:
            continue
        m = set(p["recordings"])
        ev = [(s, a, b) for a, b, s in place_edges if a in m and b in m and site[a] != site[b]]
        s, a, b = max(ev)
        out.append({"place": p["place"], "sites": p["sites"], "accounts": p["accounts"],
                    "evidence": f"{a} ({site[a]}) and {b} ({site[b]}) share {s:.0%} of views"})
    return out


def account_links(recs, place_of):
    """Accounts that share a phone, and accounts that share a place on different phones."""
    by_acct = defaultdict(list)
    for r in recs:
        by_acct[r["account"]].append(r)
    accts = sorted(by_acct)
    same_device, shared_place = [], []
    for i, a in enumerate(accts):
        for b in accts[i + 1:]:
            d = min(device_distance(x["calib"], y["calib"]) for x in by_acct[a] for y in by_acct[b])
            pa = {place_of[x["id"]] for x in by_acct[a]}
            pb = {place_of[y["id"]] for y in by_acct[b]}
            common = sorted(pa & pb)
            if d < SAME_DEVICE:
                same_device.append({"accounts": [a, b], "distance": round(d, 2), "places": common,
                                    "recordings": len(by_acct[a]) + len(by_acct[b]),
                                    "action": "REVIEW: one phone on two accounts (rule: one account per person)"})
            elif common:
                shared_place.append({"accounts": [a, b], "distance": round(d, 1), "places": common,
                                     "action": "OK: different phones in one place (household or coworkers); "
                                               "counts as one place for diversity caps"})
    return same_device, shared_place
