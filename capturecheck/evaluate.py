"""Score a pipeline result against the generator's ground truth, plus two baselines."""
import hashlib
from itertools import combinations

import numpy as np

from .dupes import Signatures
from .links import UnionFind
from .media import hamming_matrix


def _dup_truth(truth):
    uf = UnionFind(sorted({x for d in truth["dups"] for x in (d["orig"], d["copy"])}))
    for d in truth["dups"]:
        uf.union(d["orig"], d["copy"])
    return {tuple(sorted(p)) for g in uf.groups() for p in combinations(g, 2)}


def _prf(found, truth):
    tp = len(found & truth)
    return {"tp": tp, "fp": len(found - truth), "fn": len(truth - found)}


def score(res, truth):
    rows = {r["id"]: r for r in res["recordings"]}
    out = {}
    out["duplicates"] = _prf({tuple(sorted((d["orig"], d["copy"]))) for d in res["duplicates"]}, _dup_truth(truth))
    copies = {d["copy"] for d in truth["dups"]}
    out["copies_held"] = {"held": sum(rows[c]["status"] == "DUPLICATE" for c in copies), "of": len(copies)}

    rej = {(i, code) for i, r in rows.items() for code, _ in r["reasons"]}
    out["quality"] = _prf(rej, set(truth["quality"].items()))
    idle_found = {i for i, r in rows.items() if r["idle"] and r["status"] != "REJECT"}
    out["idle"] = _prf(idle_found, set(truth["idle"]))
    out["idle_err_s"] = max([abs(sum(b - a for a, b in rows[i]["idle"]) - s)
                             for i, s in truth["idle"].items() if i in idle_found] or [0.0])

    orig = [i for i in rows if i not in copies]
    same_true = {(a, b) for a, b in combinations(sorted(orig), 2) if truth["env"][a] == truth["env"][b]}
    same_found = {(a, b) for a, b in combinations(sorted(orig), 2) if rows[a]["place"] == rows[b]["place"]}
    out["same_place_pairs"] = _prf(same_found, same_true)
    out["site_aliases"] = _prf({frozenset(a["sites"]) for a in res["site_aliases"]},
                               {frozenset(s) for s in truth["site_alias"]})
    out["same_device"] = _prf({frozenset(l["accounts"]) for l in res["same_device"]},
                              {frozenset(s) for s in truth["multi_account"]})
    naive = sum(r["naive_paid_usd"] for r in rows.values())
    paid = sum(r["paid_usd"] for r in rows.values())
    out["money"] = {"naive_usd": round(naive, 2), "checked_usd": round(paid, 2),
                    "dup_usd": round(sum(min(r["dup_overlap_s"], r["payable_s"]) / 3600 * r["rate"]
                                         for r in rows.values()), 2),
                    "idle_usd": round(sum(sum(b - a for a, b in r["idle"]) / 3600 * r["rate"]
                                          for r in rows.values() if r["status"] != "REJECT"), 2)}
    return out


def baselines(recs, truth, res):
    """File hash, and 'many matching frames = duplicate' on the same candidate pairs."""
    dt = _dup_truth(truth)
    digest = {}
    for r in recs:
        digest.setdefault(hashlib.sha256(r["frames"].tobytes()).hexdigest(), []).append(r["id"])
    fh = {tuple(sorted(p)) for ids in digest.values() for p in combinations(ids, 2)}
    sig = {r["id"]: Signatures(r) for r in recs}
    naive = set()
    for key in res["place_shares"]:
        a, b = key.split("|")
        x, y = sig[a], sig[b]
        share = max((hamming_matrix(x.q, y.all).min(1) <= 8).mean(), (hamming_matrix(y.q, x.all).min(1) <= 8).mean())
        if share >= 0.5:
            naive.add(tuple(sorted((a, b))))
    return {"file_hash": _prf(fh, dt), "frame_overlap": _prf(naive, dt)}


def add(total, part):
    for k, v in part.items():
        if isinstance(v, dict):
            add(total.setdefault(k, {}), v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            total[k] = (max if k.endswith("err_s") else (lambda a, b: a + b))(total.get(k, 0), v)
    return total
