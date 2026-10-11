"""Run every check on a batch and settle what is paid and what can be delivered."""
import time

from . import links, manifest, quality
from .dupes import PLACE_SHARE, Signatures, align, candidates, place_share
from .links import UnionFind

DEFAULT_SOW = {"name": "household + workplace manipulation, egocentric", "hours": 1.5,
               "place_cap": 0.20, "person_cap": 0.15}


def run(recs, sow=DEFAULT_SOW):
    t0 = time.time()
    by_id = {r["id"]: r for r in recs}
    q = {r["id"]: quality.check(r) for r in recs}

    sigs = [Signatures(r) for r in recs]
    sig = {s.id: s for s in sigs}
    stats = {}
    pairs = candidates(sigs, stats)
    n = len(recs)
    stats.update(recordings=n, all_pairs=n * (n - 1) // 2, candidate_pairs=len(pairs))

    dups, shares = [], {}
    for a, b in pairs:
        o, c = sorted((a, b), key=lambda i: by_id[i]["uploaded_at"])
        al = align(sig[o], sig[c])
        if al:
            dups.append({"orig": o, "copy": c, **{k: round(v, 3) if isinstance(v, float) else v
                                                   for k, v in al.items()},
                         "same_account": by_id[o]["account"] == by_id[c]["account"]})
        shares[(a, b)] = place_share(sig[a], sig[b])

    # later recording of each duplicate pair loses the overlapping seconds
    overlap = {}
    for d in dups:
        overlap[d["copy"]] = max(overlap.get(d["copy"], 0.0), d["overlap_s"])
    copies = {c for c, s in overlap.items() if s >= 0.8 * by_id[c]["duration_s"]}

    edges = [(a, b, s) for (a, b), s in shares.items() if a not in copies and b not in copies]
    originals = [r for r in recs if r["id"] not in copies]
    place_list = links.places(originals, edges, PLACE_SHARE)
    place_of = {m: p["place"] for p in place_list for m in p["recordings"]}
    for d in dups:  # a re-upload belongs to the place of its original
        if d["copy"] in copies:
            place_of[d["copy"]] = place_of.get(d["orig"], place_of.get(d["copy"]))
    same_device, shared_place = links.account_links(originals, place_of)
    aliases = links.site_aliases(place_list, edges, originals)

    uf = UnionFind(sorted({r["account"] for r in recs}))
    for l in same_device:
        uf.union(*l["accounts"])
    person_of_acct = {a: "+".join(g) for g in uf.groups() for a in g}

    rows = []
    for r in recs:
        qr = q[r["id"]]
        paid_s = max(0.0, qr["payable_s"] - overlap.get(r["id"], 0.0))
        status = qr["status"]
        if r["id"] in copies:
            status = "DUPLICATE"
        elif status != "REJECT" and overlap.get(r["id"]):
            status = "PARTIAL"
        naive_s = 0.0 if qr["status"] == "REJECT" else r["duration_s"]
        rows.append({"id": r["id"], "account": r["account"], "site": r["site"], "place": place_of[r["id"]],
                     "status": status, "reasons": qr["reasons"], "idle": qr["idle"],
                     "duration_s": r["duration_s"], "payable_s": qr["payable_s"], "paid_s": paid_s,
                     "dup_overlap_s": overlap.get(r["id"], 0.0), "rate": r["rate"], "uploaded_at": r["uploaded_at"],
                     "naive_paid_usd": naive_s / 3600 * r["rate"], "paid_usd": paid_s / 3600 * r["rate"]})

    # SOW manifests: same picker, naive keys vs detected keys
    fifo = sorted(rows, key=lambda x: x["uploaded_at"])
    naive_items = [{"id": x["id"], "seconds": x["duration_s"], "env": x["site"], "person": x["account"]}
                   for x in fifo if x["status"] != "REJECT"]
    real_items = [{"id": x["id"], "seconds": x["paid_s"], "env": x["place"],
                   "person": person_of_acct[x["account"]]}
                  for x in fifo if x["status"] in ("PASS", "PARTIAL") and x["paid_s"] > 0]
    args = (sow["hours"], sow["place_cap"], sow["person_cap"])
    naive_pick, real_pick = manifest.pick(naive_items, *args), manifest.pick(real_items, *args)
    useful = {x["id"]: x["paid_s"] for x in rows}  # new, payable footage only
    person_of = {x["id"]: person_of_acct[x["account"]] for x in rows}
    lim = sow["hours"] * 3600
    man = {"sow": sow,
           "naive": {"ids": naive_pick, **manifest.audit(naive_pick, useful, place_of, person_of, lim, copies)},
           "checked": {"ids": real_pick, **manifest.audit(real_pick, useful, place_of, person_of, lim, copies)}}
    man["naive"]["claimed_places"] = len({by_id[i]["site"] for i in naive_pick})
    man["naive"]["claimed_hours"] = round(sum(by_id[i]["duration_s"] for i in naive_pick) / 3600, 2)

    stats["seconds"] = round(time.time() - t0, 1)
    return {"recordings": rows, "duplicates": dups, "places": place_list, "site_aliases": aliases,
            "same_device": same_device, "shared_place": shared_place, "manifest": man, "stats": stats,
            "place_shares": {f"{a}|{b}": round(s, 3) for (a, b), s in shares.items()}}
