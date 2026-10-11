"""Pick recordings for a client SOW under diversity caps.

Hub parses "Modality, volume, diversity caps" from the SOW (https://hub.xyz). A cap only means
something if it is counted on real places and real people. The same greedy picker runs twice:
once on claimed site IDs and accounts (naive), once on detected places and linked accounts.
"""
from collections import defaultdict


def pick(items, hours, env_cap, person_cap):
    """items: [{id, seconds, env, person}], best first. Returns the chosen ids."""
    limit = hours * 3600
    env_s, person_s = defaultdict(float), defaultdict(float)
    total, chosen = 0.0, []
    for it in items:
        if total >= limit:
            break
        s = it["seconds"]
        if env_s[it["env"]] + s > env_cap * limit or person_s[it["person"]] + s > person_cap * limit:
            continue
        chosen.append(it["id"]); total += s
        env_s[it["env"]] += s; person_s[it["person"]] += s
    return chosen


def audit(chosen, seconds, env_of, person_of, limit_s, dup_ids):
    """Measure a manifest on detected places/people and real (payable) seconds."""
    env_s, person_s = defaultdict(float), defaultdict(float)
    tot = 0.0
    for c in chosen:
        s = seconds[c]
        tot += s
        env_s[env_of[c]] += s
        person_s[person_of[c]] += s
    top_env = max(env_s.items(), key=lambda kv: kv[1]) if env_s else ("-", 0)
    top_person = max(person_s.items(), key=lambda kv: kv[1]) if person_s else ("-", 0)
    return {"recordings": len(chosen), "usable_hours": round(tot / 3600, 2),
            "places": len(env_s), "top_place": top_env[0], "top_place_share": round(top_env[1] / max(tot, 1), 3),
            "top_person": top_person[0], "top_person_share": round(top_person[1] / max(tot, 1), 3),
            "duplicates_inside": sorted(set(chosen) & set(dup_ids)),
            "short_of_target_h": round(max(0.0, limit_s - tot) / 3600, 2)}
