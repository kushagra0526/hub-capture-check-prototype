"""Score capturecheck over many generated batches.

    python evaluate_seeds.py dev        # seeds 1-5, used while building
    python evaluate_seeds.py heldout    # seeds 6-25, run once after the detector was frozen
"""
import json
import sys
from pathlib import Path

from capturecheck.evaluate import add, baselines, score
from capturecheck.pipeline import run
from capturecheck.synth import make_batch

SETS = {"dev": range(1, 6), "heldout": range(6, 26)}


def main(name):
    total, per_seed = {}, {}
    for seed in SETS[name]:
        recs, truth = make_batch(seed)
        res = run(recs)
        s = score(res, truth)
        s["baselines"] = baselines(recs, truth, res)
        s["pairs"] = {"all": res["stats"]["all_pairs"], "compared": res["stats"]["candidate_pairs"]}
        per_seed[seed] = s
        add(total, s)
        print(seed, json.dumps({k: v for k, v in s.items() if k in ("duplicates", "same_place_pairs", "quality", "idle")}),
              flush=True)
    out = Path("out") / f"eval_{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"total": total, "per_seed": per_seed}, indent=1))
    print(json.dumps(total, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
