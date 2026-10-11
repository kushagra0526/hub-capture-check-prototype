"""One contributor batch through capturecheck → out/report.html, out/report.json

    python demo.py            # batch seed 1 (synthetic)
"""
import html
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from capturecheck.dupes import Signatures, _smooth  # noqa: E402
from capturecheck.media import fmt_t, hamming_matrix  # noqa: E402
from capturecheck.pipeline import DEFAULT_SOW, run  # noqa: E402
from capturecheck.synth import make_batch  # noqa: E402

SEED = 1


def _pix(frame):
    return (np.clip(frame, 0, 1) * 255).astype(int).ravel().tolist()


def _profiles(orig, copy, d):
    a, b = _smooth(orig.profile), _smooth(copy.profile)
    j = np.arange(len(b))
    src = d["rate"] * j + d["offset_s"] * 4
    ok = (src >= 0) & (src <= len(a) - 1)
    pa, pb = np.interp(src[ok], np.arange(len(a)), a), b[ok]
    z = lambda x: ((x - x.mean()) / (x.std() + 1e-9)).round(2)[::3].tolist()
    return z(pa), z(pb)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    recs, truth = make_batch(SEED)
    res = run(recs)
    by_id = {r["id"]: r for r in recs}
    rows = {r["id"]: r for r in res["recordings"]}
    sig = {r["id"]: Signatures(r) for r in recs}

    naive_usd = sum(r["naive_paid_usd"] for r in rows.values())
    paid_usd = sum(r["paid_usd"] for r in rows.values())
    dup_usd = sum(min(r["dup_overlap_s"], r["payable_s"]) / 3600 * r["rate"] for r in rows.values())
    idle_usd = sum(sum(b - a for a, b in r["idle"]) / 3600 * r["rate"] for r in rows.values() if r["status"] != "REJECT")
    rejected = [r for r in rows.values() if r["status"] == "REJECT"]
    partial = [r for r in rows.values() if r["status"] == "PARTIAL"]
    accounts = {r["account"] for r in recs}

    print(f"\n== Batch (synthetic, seed {SEED}): {len(recs)} uploads from {len(accounts)} accounts, "
          f"{sum(r['duration_s'] for r in recs) / 3600:.2f} h")
    print(f"   compared {res['stats']['candidate_pairs']} of {res['stats']['all_pairs']} recording pairs "
          f"(LSH), {res['stats']['seconds']} s")
    print("\n-- Re-uploads (held, overlap not paid again)")
    for d in res["duplicates"]:
        tr = next((t for t in truth["dups"] if {t["orig"], t["copy"]} == {d["orig"], d["copy"]}), None)
        print(f"   {d['copy']} repeats {d['orig']}: speed x{d['rate']:.2f}, starts at {fmt_t(d['offset_s'])} "
              f"of the original, {fmt_t(d['overlap_s'])} overlap, motion match {d['ncc']:.2f}"
              f"{' (same account)' if d['same_account'] else ' (different account)'}"
              f"   [planted: {tr['how'] if tr else 'copy of a copy'}]")
    print("\n-- Rejected, with the reason the contributor sees")
    for r in rejected:
        for code, why in r["reasons"]:
            print(f"   {r['id']} {code}: {why}")
    print("\n-- Paid in part (idle with no hands and a still phone is not paid)")
    for r in partial:
        if r["idle"]:
            print(f"   {r['id']}: unpaid {', '.join(f'{fmt_t(a)}-{fmt_t(b)}' for a, b in r['idle'])}")
    print("\n-- Places: one physical place, several site IDs")
    for a in res["site_aliases"]:
        print(f"   {a['place']}: {', '.join(a['sites'])} ({a['evidence']})")
    print("\n-- Accounts")
    for l in res["same_device"]:
        print(f"   {' + '.join(l['accounts'])}: same phone calibration (distance {l['distance']}) → {l['action']}")
    for l in res["shared_place"]:
        print(f"   {' + '.join(l['accounts'])}: {l['action']}")
    m = res["manifest"]
    print(f"\n-- SOW: {DEFAULT_SOW['hours']} h, max {DEFAULT_SOW['place_cap']:.0%} per place, "
          f"max {DEFAULT_SOW['person_cap']:.0%} per person")
    print(f"   naive (site IDs, accounts): claims {m['naive']['claimed_places']} places; really {m['naive']['places']}, "
          f"top place {m['naive']['top_place_share']:.0%}, {len(m['naive']['duplicates_inside'])} re-uploads inside")
    print(f"   checked (places, linked accounts): {m['checked']['places']} places, top place "
          f"{m['checked']['top_place_share']:.0%}, top person {m['checked']['top_person_share']:.0%}, 0 re-uploads")
    print(f"\n-- Payout at Hub's published rates: per-recording review ${naive_usd:.2f} → capturecheck ${paid_usd:.2f} "
          f"(re-uploads ${dup_usd:.2f}, idle ${idle_usd:.2f})")

    # ---- visuals for the report
    combo = next(d for d in res["duplicates"]
                 if next((t["how"] for t in truth["dups"] if t["copy"] == d["copy"]), "") == "combo")
    o, c = by_id[combo["orig"]], by_id[combo["copy"]]
    js = np.linspace(len(c["frames"]) * 0.12, len(c["frames"]) * 0.85, 4).astype(int)
    strip = [{"copy": _pix(c["frames"][j]), "orig": _pix(o["frames"][int(combo["rate"] * j + combo["offset_s"] * 4)]),
              "t_copy": fmt_t(j / 4), "t_orig": fmt_t((combo["rate"] * j + combo["offset_s"] * 4) / 4)} for j in js]
    prof_o, prof_c = _profiles(sig[o["id"]], sig[c["id"]], combo)

    copies = {d["copy"] for d in res["duplicates"]}
    same = [(s, k) for k, s in res["place_shares"].items()
            if all(x not in copies for x in k.split("|"))
            and by_id[k.split("|")[0]]["account"] == by_id[k.split("|")[1]]["account"]]
    s_share, key = max(same)
    a, b = key.split("|")
    neg = {"a": a, "b": b, "share": s_share, "account": by_id[a]["account"],
           "frames": [{"a": _pix(by_id[a]["frames"][i]), "b": _pix(by_id[b]["frames"][i])} for i in (40, 200, 360)],
           "motion": float(np.corrcoef(_smooth(sig[a].profile)[:400], _smooth(sig[b].profile)[:400])[0, 1]),
           "naive_says_copy": bool(max((hamming_matrix(sig[a].q, sig[b].all).min(1) <= 8).mean(),
                                       (hamming_matrix(sig[b].q, sig[a].all).min(1) <= 8).mean()) >= 0.5)}

    heldout = json.loads((HERE / "out" / "eval_heldout.json").read_text())["total"] \
        if (HERE / "out" / "eval_heldout.json").exists() else None

    data = {"seed": SEED, "uploads": len(recs), "held": sum(r["status"] == "DUPLICATE" for r in rows.values()), "accounts": len(accounts), "stats": res["stats"],
            "duplicates": res["duplicates"], "rejected": [{"id": r["id"], "reasons": r["reasons"]} for r in rejected],
            "idle": [{"id": r["id"], "idle": r["idle"]} for r in partial if r["idle"]],
            "site_aliases": res["site_aliases"], "same_device": res["same_device"], "shared_place": res["shared_place"],
            "manifest": {k: {kk: vv for kk, vv in v.items() if kk != "ids"} if isinstance(v, dict) and "ids" in v else v
                         for k, v in m.items()},
            "money": {"naive_usd": round(naive_usd, 2), "paid_usd": round(paid_usd, 2),
                      "dup_usd": round(dup_usd, 2), "idle_usd": round(idle_usd, 2)},
            "heldout": heldout}
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "report.json").write_text(json.dumps(data, indent=1, default=float))
    (HERE / "out" / "report.html").write_text(render(data, strip, prof_o, prof_c, neg, combo), encoding="utf-8")
    print("\nwrote out/report.html, out/report.json")
    return 0


def render(d, strip, prof_o, prof_c, neg, combo):
    e = html.escape
    m = d["manifest"]
    ho = d["heldout"]

    def poly(vals, w=520, h=90, lo=-2.5, hi=3.5):
        n = len(vals)
        return " ".join(f"{i * w / (n - 1):.1f},{h - (min(max(v, lo), hi) - lo) / (hi - lo) * h:.1f}" for i, v in enumerate(vals))

    rej = "".join(f"<tr><td>{e(r['id'])}</td><td><b>{e(c)}</b></td><td>{e(w)}</td></tr>"
                  for r in d["rejected"] for c, w in r["reasons"])
    idle = "".join(f"<tr><td>{e(r['id'])}</td><td><b>IDLE</b></td><td>not paid: "
                   f"{', '.join(fmt_t(a) + '–' + fmt_t(b) for a, b in r['idle'])} (no hands, phone still)</td></tr>"
                   for r in d["idle"])
    dups = "".join(f"<tr><td>{e(x['copy'])}</td><td>{e(x['orig'])}</td><td>×{x['rate']:.2f}</td>"
                   f"<td>{fmt_t(x['offset_s'])}</td><td>{fmt_t(x['overlap_s'])}</td><td>{x['ncc']:.2f}</td>"
                   f"<td>{'same' if x['same_account'] else 'other'}</td></tr>" for x in d["duplicates"])
    alias = "".join(f"<li><b>{e(', '.join(a['sites']))}</b> are one place. {e(a['evidence'])}</li>" for a in d["site_aliases"])
    link = "".join(f"<li><b>{e(' + '.join(l['accounts']))}</b>: same phone calibration → review (one account per person)</li>"
                   for l in d["same_device"])
    link += "".join(f"<li>{e(' + '.join(l['accounts']))}: different phones, same place → OK, counted as one place</li>"
                    for l in d["shared_place"])
    ho_html = ""
    if ho:
        b = ho["baselines"]
        ho_html = f"""
<section><h2>Held-out: 20 fresh batches (seeds 6–25), run once after the detector was frozen in git</h2>
<table class="num"><tr><th></th><th>caught</th><th>false flags</th><th>missed</th></tr>
<tr><td>Re-upload pairs</td><td>{ho['duplicates']['tp']}</td><td>{ho['duplicates']['fp']}</td><td>{ho['duplicates']['fn']}</td></tr>
<tr><td>Rejects (shake / dark / no hands)</td><td>{ho['quality']['tp']}</td><td>{ho['quality']['fp']}</td><td>{ho['quality']['fn']}</td></tr>
<tr><td>Idle padding</td><td>{ho['idle']['tp']}</td><td>{ho['idle']['fp']}</td><td>{ho['idle']['fn']}</td></tr>
<tr><td>Same-place pairs</td><td>{ho['same_place_pairs']['tp']}</td><td>{ho['same_place_pairs']['fp']}</td><td>{ho['same_place_pairs']['fn']}</td></tr>
<tr><td>One phone, two accounts</td><td>{ho['same_device']['tp']}</td><td>{ho['same_device']['fp']}</td><td>{ho['same_device']['fn']}</td></tr>
<tr class="base"><td>Baseline: file hash</td><td>{b['file_hash']['tp']}</td><td>{b['file_hash']['fp']}</td><td>{b['file_hash']['fn']}</td></tr>
<tr class="base"><td>Baseline: "many matching frames = copy"</td><td>{b['frame_overlap']['tp']}</td><td>{b['frame_overlap']['fp']}</td><td>{b['frame_overlap']['fn']}</td></tr>
</table>
<p class="note">Pairs compared: {ho['pairs']['compared']:,} of {ho['pairs']['all']:,} (LSH). Same generator, unseen random draws; the hard cases were written by me.</p></section>"""

    return f"""<!doctype html><html><head><meta charset="utf-8"><title>capturecheck report</title>
<style>
:root{{--bg:#0f1115;--card:#171a21;--ink:#e8eaf0;--mute:#9aa3b2;--line:#2a2f3a;--good:#4ade80;--bad:#f87171;--warn:#fbbf24;--acc:#60a5fa}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:28px 16px}}
h1{{margin:0 0 4px;font-size:26px}} h2{{font-size:17px;margin:0 0 10px}}
.sub{{color:var(--mute);margin-bottom:18px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:16px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}}
.kpi b{{display:block;font-size:24px}} .kpi b span{{font-size:24px;color:var(--good)}} .kpi span{{color:var(--mute);font-size:13px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:14px}}
table{{border-collapse:collapse;width:100%;font-size:14px}} td,th{{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left}}
th{{color:var(--mute);font-weight:500}} .num td:not(:first-child),.num th:not(:first-child){{text-align:right}}
.base td{{color:var(--mute)}} .note{{color:var(--mute);font-size:13px;margin:8px 0 0}}
.strip{{display:flex;gap:10px;flex-wrap:wrap}} .pair{{text-align:center;font-size:12px;color:var(--mute)}}
canvas{{width:128px;height:96px;image-rendering:pixelated;border-radius:4px;display:block;margin:2px auto}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} @media(max-width:800px){{.two{{grid-template-columns:1fr}}}}
.good{{color:var(--good)}} .bad{{color:var(--bad)}} ul{{margin:4px 0;padding-left:18px}}
svg{{width:100%;height:auto;background:#11141a;border-radius:6px}}
</style></head><body><main>
<h1>capturecheck: review before payout and before delivery</h1>
<div class="sub">Synthetic batch (seed {d['seed']}): {d['uploads']} uploads from {d['accounts']} accounts. Every result below is computed, not typed.
Unofficial prototype by Kushagra Pratap Singh. Not affiliated with Hub.</div>
<div class="kpis">
<div class="kpi"><b class="bad">{d['held']}</b><span>re-uploads held (crop, mirror, speed, trim)</span></div>
<div class="kpi"><b>{len(d['rejected'])}</b><span>rejected, each with a timestamped reason</span></div>
<div class="kpi"><b>{len(d['site_aliases'])}</b><span>places registered under several site IDs</span></div>
<div class="kpi"><b>{len(d['same_device'])}</b><span>phone on two accounts → review</span></div>
<div class="kpi"><b>${d['money']['naive_usd']:.2f} → <span class="good">${d['money']['paid_usd']:.2f}</span></b><span>batch payout at Hub's published rates</span></div>
<div class="kpi"><b>{d['stats']['candidate_pairs']}/{d['stats']['all_pairs']}</b><span>pairs compared (LSH index)</span></div>
</div>
<div class="two">
<section><h2>Caught: a mirrored, cropped, sped-up, trimmed copy</h2>
<div class="strip">{''.join(f'<div class="pair"><canvas data-px="{i}" data-k="orig" width="32" height="24"></canvas>orig {s["t_orig"]}<canvas data-px="{i}" data-k="copy" width="32" height="24"></canvas>copy {s["t_copy"]}</div>' for i, s in enumerate(strip))}</div>
<svg viewBox="0 0 520 90"><polyline fill="none" stroke="#60a5fa" stroke-width="1.6" points="{poly(prof_o)}"/><polyline fill="none" stroke="#fbbf24" stroke-width="1.2" points="{poly(prof_c)}"/></svg>
<p class="note">Hand and head motion of the copy (yellow) on top of the original (blue) after the fitted time map. {e(combo['copy'])} repeats {e(combo['orig'])} at ×{combo['rate']:.2f} speed; motion matches at {combo['ncc']:.2f}.</p></section>
<section><h2>Not flagged: same person, same kitchen, a different task</h2>
<div class="strip">{''.join(f'<div class="pair"><canvas data-neg="{i}" data-k="a" width="32" height="24"></canvas>{e(neg["a"])}<canvas data-neg="{i}" data-k="b" width="32" height="24"></canvas>{e(neg["b"])}</div>' for i in range(3))}</div>
<p class="note">{e(neg['account'])}: the backgrounds match ({neg['share']:.0%} of views verified) so they are <b>one place</b> for diversity caps.
The motion does not repeat (correlation {neg['motion']:.2f}), so it is <b class="good">not a copy</b>. {'The "many matching frames" rule calls this a duplicate.' if neg['naive_says_copy'] else ''}</p></section>
</div>
<section><h2>Re-uploads held (every matching pair; a copy of a copy shows twice)</h2><table class="num"><tr><th>upload</th><th>repeats</th><th>speed</th><th>starts at</th><th>overlap</th><th>motion</th><th>account</th></tr>{dups}</table></section>
<section><h2>What the contributor is told</h2><table>{rej}{idle}</table></section>
<div class="two">
<section><h2>Places and accounts</h2><ul>{alias}</ul><ul>{link}</ul></section>
<section><h2>SOW: {m['sow']['hours']} h, ≤{m['sow']['place_cap']:.0%} per place, ≤{m['sow']['person_cap']:.0%} per person</h2>
<table class="num"><tr><th></th><th>naive (site IDs)</th><th>capturecheck</th></tr>
<tr><td>Places it claims</td><td>{m['naive']['claimed_places']}</td><td>{m['checked']['places']}</td></tr>
<tr><td>Real places</td><td>{m['naive']['places']}</td><td>{m['checked']['places']}</td></tr>
<tr><td>Top place share</td><td class="{'bad' if m['naive']['top_place_share'] > m['sow']['place_cap'] else ''}">{m['naive']['top_place_share']:.0%}</td><td class="good">{m['checked']['top_place_share']:.0%}</td></tr>
<tr><td>Re-uploads inside</td><td class="{'bad' if m['naive']['duplicates_inside'] else ''}">{len(m['naive']['duplicates_inside'])}</td><td>0</td></tr>
<tr><td>New footage (h)</td><td>{m['naive']['usable_hours']} of {m['naive']['claimed_hours']} claimed</td><td>{m['checked']['usable_hours']}</td></tr></table></section>
</div>
{ho_html}
</main><script>
const S={json.dumps(strip)},N={json.dumps(neg['frames'])};
function draw(c,px){{const x=c.getContext('2d'),im=x.createImageData(32,24);for(let i=0;i<px.length;i++){{im.data[4*i]=im.data[4*i+1]=im.data[4*i+2]=px[i];im.data[4*i+3]=255}}x.putImageData(im,0,0)}}
document.querySelectorAll('canvas[data-px]').forEach(c=>draw(c,S[c.dataset.px][c.dataset.k]));
document.querySelectorAll('canvas[data-neg]').forEach(c=>draw(c,N[c.dataset.neg][c.dataset.k]));
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
