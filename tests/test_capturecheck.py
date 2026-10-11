import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capturecheck import quality  # noqa: E402
from capturecheck.dupes import Signatures, align, place_share  # noqa: E402
from capturecheck.evaluate import score  # noqa: E402
from capturecheck.media import dhash, hamming_matrix  # noqa: E402
from capturecheck.pipeline import run  # noqa: E402
from capturecheck.synth import Device, Env, make_batch, make_copy, record  # noqa: E402


@pytest.fixture(scope="module")
def batch():
    recs, truth = make_batch(1)
    return recs, truth, run(recs)


def test_every_planted_problem_found_with_no_false_flags(batch):
    recs, truth, res = batch
    s = score(res, truth)
    for k in ("duplicates", "quality", "idle", "same_place_pairs", "site_aliases", "same_device"):
        assert s[k]["fp"] == 0 and s[k]["fn"] == 0, (k, s[k])
    assert s["copies_held"]["held"] == s["copies_held"]["of"] == 6


def test_each_transform_is_recovered_with_its_speed():
    rng = np.random.default_rng(3)
    env, dev, other = Env(rng, "e", "home"), Device(rng, "d1"), Device(rng, "d2")
    src = record(rng, "a", env, dev, "acct-1", "H-1", 3)
    for how in ("reencode", "crop", "mirror", "speed", "trim", "combo"):
        cp, info = make_copy(rng, src, "b", "acct-2", "H-2", other, how, 10)
        al = align(Signatures(src), Signatures(cp))
        assert al is not None, how
        assert abs(al["rate"] - info.get("speed", 1.0)) < 0.02, (how, al, info)


def test_same_room_different_task_is_one_place_but_not_a_copy():
    rng = np.random.default_rng(4)
    env, dev = Env(rng, "kitchen", "home"), Device(rng, "d")
    a = record(rng, "a", env, dev, "acct-1", "H-1", 3, static=True)
    b = record(rng, "b", env, dev, "acct-1", "H-1", 3, static=True)
    sa, sb = Signatures(a), Signatures(b)
    assert place_share(sa, sb) > 0.3
    assert align(sa, sb) is None
    # the naive rule would have called it a copy
    assert (hamming_matrix(sa.q, sb.all).min(1) <= 8).mean() >= 0.5


def test_mirror_breaks_plain_hash_but_not_the_check():
    rng = np.random.default_rng(5)
    env, dev = Env(rng, "e", "home"), Device(rng, "d")
    src = record(rng, "a", env, dev, "acct-1", "H-1", 2)
    cp, _ = make_copy(rng, src, "b", "acct-1", "H-1", dev, "mirror", 5)
    plain = hamming_matrix(dhash(cp["frames"][::4]), dhash(src["frames"])).min(1)
    assert np.median(plain) > 8
    assert align(Signatures(src), Signatures(cp))["ncc"] > 0.9


def test_reject_reasons_have_timestamps_and_short_pauses_are_paid():
    rng = np.random.default_rng(6)
    env, dev = Env(rng, "e", "home"), Device(rng, "d")
    shaky = quality.check(record(rng, "s", env, dev, "a", "H", 2, shaky=True))
    assert shaky["status"] == "REJECT" and shaky["reasons"][0][0] == "STEADY" and "-" in shaky["reasons"][0][1]
    ok = quality.check(record(rng, "o", env, dev, "a", "H", 2))
    assert ok["status"] == "PASS" and ok["payable_s"] == pytest.approx(120, abs=1)
    padded = record(rng, "p", env, dev, "a", "H", 2, idle=(0.5, 90))
    q = quality.check(padded)
    assert q["status"] == "PARTIAL" and abs(q["idle_s"] - 90) < 12


def test_naive_manifest_breaks_the_place_cap_and_checked_does_not(batch):
    _, _, res = batch
    m = res["manifest"]
    cap = m["sow"]["place_cap"]
    assert m["naive"]["claimed_places"] > m["naive"]["places"]
    assert m["naive"]["top_place_share"] > cap
    assert m["checked"]["top_place_share"] <= cap + 1e-9
    assert not m["checked"]["duplicates_inside"]


def test_lsh_compares_a_fraction_of_pairs(batch):
    _, _, res = batch
    assert res["stats"]["candidate_pairs"] < 0.25 * res["stats"]["all_pairs"]
