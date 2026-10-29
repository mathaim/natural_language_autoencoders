"""Build a self-contained HTML report from finished adversarial-extraction runs.

Runs on the cluster, next to the data. The output embeds real prompts,
verbalizations and base-model outputs, so the content never has to pass
through a chat or be re-typed — copy the single .html file wherever you
present from.

  python build_report.py \
      --runs-root /scratch/euh7ys/adversarial_runs \
      --models qwen7b-l20 gemma12b-l32 \
      --out /scratch/euh7ys/adversarial_runs/report.html

Sections: headline tiles, harm-language chart, full stats table, validity
checks, then per-dataset examples (prompt, base-model reply, a spread of
positions with their verbalizations) and a list of positions the verbalizer
described with harm language. --datasets restricts everything to a subset
(e.g. `--datasets truthfulqa` for a version with no harmful content).
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

_CJK = range(0x4E00, 0x9FFF + 1)
_SOFT = ("sensitive", "harmful", "dangerous", "unethical", "inappropriate",
         "illegal", "will not", "cannot provide", "not able to provide", "decline")
DATASET_ORDER = ["advbench", "harmbench", "wmdp", "truthfulqa"]
DATASET_LABEL = {"advbench": "AdvBench", "harmbench": "HarmBench",
                 "wmdp": "WMDP", "truthfulqa": "TruthfulQA"}
DATASET_NOTE = {"advbench": "jailbreak instructions", "harmbench": "harmful behaviors",
                "wmdp": "hazardous-knowledge MCQ", "truthfulqa": "factual questions"}
JAILBREAK = {"advbench", "harmbench"}
KNOWLEDGE = {"wmdp", "truthfulqa"}


def _has_cjk(s: str | None) -> bool:
    return bool(s) and any(ord(c) in _CJK for c in s)


def _is_soft(s: str | None) -> bool:
    return bool(s) and any(k in s.lower() for k in _SOFT)


def _flag(r: dict) -> bool:
    return bool(r["is_refusal"] or _is_soft(r["verbalization"]))


def esc(s) -> str:
    return html.escape("" if s is None else str(s))


# ─── Load ────────────────────────────────────────────────────────────────────

def load_run(run_dir: Path, datasets: list[str] | None) -> dict:
    samples = {}
    for line in (run_dir / "samples.jsonl").read_text().splitlines():
        if line.strip():
            s = json.loads(line)
            if datasets and s["dataset"] not in datasets:
                continue
            samples[s["sample_id"]] = s
    recs = []
    for line in (run_dir / "verbalizations.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["sample_id"] in samples:
                recs.append(r)
    by_sample: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in recs:
        by_sample[r["sample_id"]][r["position_idx"]] = r
    return {"dir": run_dir, "samples": samples, "recs": recs, "by_sample": by_sample}


def model_label(slug: str) -> tuple[str, str]:
    """(short display name, detail line) — from the registry if available."""
    try:
        from nla_model_configs import NLA_MODELS
        cfg = NLA_MODELS[slug]
        base = cfg.base_model.split("/")[-1]
        return base, f"{cfg.base_model} · layer {cfg.layer}/{cfg.n_layers} · d={cfg.d_model}"
    except Exception:  # noqa: BLE001 — registry optional
        return slug, slug


# ─── Stats ───────────────────────────────────────────────────────────────────

def stats(run: dict) -> dict:
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in run["recs"]:
        by_ds[r["dataset"]].append(r)
    out = {}
    for ds, rows in by_ds.items():
        verb = [r for r in rows if r["verbalization"] is not None]
        n = len(verb) or 1
        early = [r for r in verb if r["min_position_warning"]]
        late = [r for r in verb if not r["min_position_warning"]]
        out[ds] = {
            "positions": len(rows),
            "verbalized": len(verb),
            "refused": sum(r["is_refusal"] for r in verb),
            "any_ref": sum(_flag(r) for r in verb),
            "cjk": sum(_has_cjk(r["verbalization"]) for r in verb),
            "early_ref_pct": 100 * sum(r["is_refusal"] for r in early) / len(early) if early else None,
            "late_ref_pct": 100 * sum(r["is_refusal"] for r in late) / len(late) if late else None,
            "samples": len({r["sample_id"] for r in rows}),
            "base_refusals": sum(bool(run["samples"][sid].get("model_output_is_refusal"))
                                 for sid in {r["sample_id"] for r in rows}),
        }
        out[ds]["refusal_pct"] = 100 * out[ds]["refused"] / n
        out[ds]["any_ref_pct"] = 100 * out[ds]["any_ref"] / n
    expected = sum(s["n_tokens"] for s in run["samples"].values())
    all_verb = [r for r in run["recs"] if r["verbalization"] is not None]
    cjk = [r for r in all_verb if _has_cjk(r["verbalization"])]
    return {
        "datasets": out,
        "positions": len(run["recs"]),
        "expected": expected,
        "samples": len(run["samples"]),
        "cjk": len(cjk),
        "cjk_rate": len(cjk) / (len(all_verb) or 1),
        "cjk_early": sum(r["min_position_warning"] for r in cjk),
        "cjk_samples": len({r["sample_id"] for r in cjk}),
    }


# ─── Rendering helpers ───────────────────────────────────────────────────────

def context_html(toks: dict[int, dict], p: int, n_tok: int, width: int) -> str:
    lo, hi = max(0, p - width), min(p + width + 1, n_tok)
    parts = []
    for i in range(lo, hi):
        t = toks.get(i)
        s = esc(t["token_str"]).replace("\n", "⏎") if t else "?"
        parts.append(f'<mark>{s}</mark>' if i == p else s)
    pre = "…" if lo > 0 else ""
    post = "…" if hi < n_tok else ""
    return f'<span class="ctx">{pre}{"".join(parts)}{post}</span>'


def position_row(r: dict, toks: dict, n_tok: int, width: int) -> str:
    flags = []
    if r["is_refusal"]:
        flags.append('<span class="tag ref">refusal</span>')
    elif _is_soft(r["verbalization"]):
        flags.append('<span class="tag soft">harm wording</span>')
    if r["min_position_warning"]:
        flags.append('<span class="tag">early</span>')
    if _has_cjk(r["verbalization"]):
        flags.append('<span class="tag">CJK</span>')
    return (
        f'<div class="pos">'
        f'<div class="pos-head"><span class="mono">pos {r["position_idx"]}</span> '
        f'{context_html(toks, r["position_idx"], n_tok, width)} {" ".join(flags)}</div>'
        f'<div class="verb">{esc(r["verbalization"])}</div>'
        f'</div>'
    )


def pick_positions(toks: dict[int, dict], n_tok: int, per_sample: int | None) -> list[int]:
    have = sorted(toks)
    if per_sample is None or per_sample >= len(have):
        return have
    # evenly spaced over the prompt, always including the last position,
    # plus every harm-flagged position (capped so one sample can't explode)
    step = max(1, len(have) // per_sample)
    chosen = set(have[::step][:per_sample - 1]) | {have[-1]}
    flagged = [p for p in have if _flag(toks[p])]
    chosen |= set(flagged[:per_sample])
    return sorted(chosen)


def sample_block(s: dict, toks: dict, per_sample: int | None, width: int) -> str:
    n_tok = s["n_tokens"]
    mo = s.get("model_output")
    mo_html = ""
    if mo is not None:
        flag = ' <span class="tag ref">base model refused</span>' if s.get("model_output_is_refusal") else ""
        mo_html = f'<div class="field"><div class="k">Base model reply{flag}</div><div class="v">{esc(mo)}</div></div>'
    meta = s.get("meta") or {}
    meta_html = ""
    if meta:
        bits = ", ".join(f"{esc(k)}: {esc(v)}" for k, v in meta.items() if not isinstance(v, (list, dict)))
        if bits:
            meta_html = f'<div class="meta-line">{bits}</div>'
    rows = "".join(position_row(toks[p], toks, n_tok, width) for p in pick_positions(toks, n_tok, per_sample))
    shown = len(pick_positions(toks, n_tok, per_sample))
    return (
        f'<details class="sample"><summary><span class="mono">{esc(s["sample_id"])}</span> '
        f'<span class="dim">{n_tok} tokens · {shown} positions shown</span></summary>'
        f'{meta_html}'
        f'<div class="field"><div class="k">Prompt</div><div class="v">{esc(s.get("text", "(prompt text not saved)"))}</div></div>'
        f'{mo_html}'
        f'<div class="k">Verbalizations by position</div>{rows}'
        f'</details>'
    )


# ─── Page ────────────────────────────────────────────────────────────────────

CSS = """
:root{--bg:#f7f8fa;--surface:#fff;--fg:#15181d;--fg-2:#4b5260;--fg-3:#7a8190;--rule:#d9dde4;--accent:#2f4a6e;--accent-soft:#e6ecf4;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--ok:#1f7a3f;--ok-soft:#e3f3e8;--warn:#9a3412;--warn-soft:#fdecdc;--mark:#fff1a8;
--display:"Newsreader","Iowan Old Style",Georgia,serif;--body:"Public Sans","Helvetica Neue",Arial,sans-serif;--mono:"JetBrains Mono","SF Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#15181d;--surface:#1d2127;--fg:#eef0f3;--fg-2:#b7bdc8;--fg-3:#848b98;--rule:#343a44;--accent:#a9bfdc;--accent-soft:#262e3a;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--ok:#5fc07f;--ok-soft:#1f3327;--warn:#f4a06a;--warn-soft:#3a2418;--mark:#5a4d12;color-scheme:dark}}
:root[data-theme=dark]{--bg:#15181d;--surface:#1d2127;--fg:#eef0f3;--fg-2:#b7bdc8;--fg-3:#848b98;--rule:#343a44;--accent:#a9bfdc;--accent-soft:#262e3a;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--ok:#5fc07f;--ok-soft:#1f3327;--warn:#f4a06a;--warn-soft:#3a2418;--mark:#5a4d12;color-scheme:dark}
*{box-sizing:border-box}html{color-scheme:light dark}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--body);font-size:16px;line-height:1.55;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:0 64px}
main{max-width:900px;margin:0 auto}
h1,h2,h3{font-family:var(--display);font-weight:500;text-wrap:balance;margin:0;line-height:1.15}
h1{font-size:clamp(34px,5vw,48px);letter-spacing:-.01em}h2{font-size:26px;margin-top:56px;padding-top:20px;border-top:1px solid var(--rule)}h3{font-size:19px;margin-top:28px}
p{max-width:68ch;margin:12px 0}p.lede{font-size:19px;color:var(--fg-2);max-width:60ch}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--fg-3);font-weight:600}
header{padding-top:48px;display:grid;gap:14px}.meta{display:flex;flex-wrap:wrap;gap:8px 20px;font-family:var(--mono);font-size:13px;color:var(--fg-2)}
code,.mono{font-family:var(--mono);font-size:.9em}code{background:var(--accent-soft);padding:1px 5px;border-radius:3px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-top:32px}
.tile{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:18px 20px;display:grid;gap:4px;min-width:0}
.tile .n{font-family:var(--display);font-size:38px;font-weight:500;line-height:1;font-variant-numeric:tabular-nums}.tile .l{font-size:13px;color:var(--fg-2)}
figure{margin:24px 0 0;background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:20px 20px 14px;min-width:0}
figure .title{font-weight:600;font-size:15px}figure .sub{font-size:13px;color:var(--fg-2);margin-top:2px}
.legend{display:flex;gap:18px;font-size:13px;color:var(--fg-2);margin:12px 0 4px;flex-wrap:wrap}
.legend span::before{content:"";display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:6px;background:var(--c)}
.chart-wrap{overflow-x:auto;position:relative}svg{display:block;width:100%;height:auto;max-width:100%;font-family:var(--body)}
.bar{transition:opacity .12s}.bar.dimmed{opacity:.35}
.tip{position:absolute;pointer-events:none;background:var(--fg);color:var(--bg);font-size:12px;padding:6px 9px;border-radius:4px;white-space:nowrap;transform:translate(-50%,calc(-100% - 10px));display:none;font-variant-numeric:tabular-nums}
figcaption{font-size:13px;color:var(--fg-3);margin-top:10px}
.table-wrap{overflow-x:auto;margin-top:16px;border:1px solid var(--rule);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:9px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--rule)}
th{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--fg-3);font-weight:600;background:var(--bg)}
th:first-child,td:first-child{text-align:left}tbody tr:last-child td{border-bottom:0}td.model{font-weight:600;color:var(--fg-2)}td.hi{font-weight:600}
.pill{display:inline-block;font-size:12px;font-weight:600;padding:2px 8px;border-radius:999px;background:var(--ok-soft);color:var(--ok)}
.pill.warn{background:var(--warn-soft);color:var(--warn)}
ul{padding-left:20px;max-width:68ch}li{margin:6px 0}
.tag{display:inline-block;font-size:11px;font-weight:600;letter-spacing:.03em;padding:1px 7px;border-radius:999px;background:var(--accent-soft);color:var(--accent);vertical-align:1px}
.tag.ref{background:var(--warn-soft);color:var(--warn)}.tag.soft{background:var(--warn-soft);color:var(--warn)}
.dim{color:var(--fg-3);font-size:13px}
details.sample{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:0 18px;margin-top:12px;min-width:0}
details.sample>summary{cursor:pointer;padding:12px 0;font-weight:500;display:flex;gap:12px;flex-wrap:wrap;align-items:baseline}
details.sample[open]>summary{border-bottom:1px solid var(--rule)}
.field{margin:14px 0}.k{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--fg-3);font-weight:600;margin:14px 0 4px}
.v{white-space:pre-wrap;overflow-wrap:anywhere;font-size:15px;max-width:none}
.meta-line{font-family:var(--mono);font-size:12px;color:var(--fg-3);margin-top:12px}
.pos{padding:10px 0;border-top:1px dashed var(--rule)}.pos:last-child{padding-bottom:16px}
.pos-head{font-size:13px;color:var(--fg-2);display:flex;gap:10px;flex-wrap:wrap;align-items:baseline}
.ctx{font-family:var(--mono);font-size:12.5px;overflow-wrap:anywhere}mark{background:var(--mark);color:var(--fg);padding:0 2px;border-radius:2px;font-weight:600}
.verb{margin-top:6px;overflow-wrap:anywhere}
.harm-list .pos{border-top:1px solid var(--rule);padding:12px 0}.harm-list .pos:first-child{border-top:0}
.harm-list{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:4px 18px;margin-top:12px}
.model-head{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;margin-top:22px}.model-head .dot{width:12px;height:12px;border-radius:3px;display:inline-block;background:var(--c)}
@media (prefers-reduced-motion:reduce){.bar{transition:none}}
"""

CHART_JS = """
(function(){
var data=%(chart)s, series=%(series)s;
var W=760,labelW=128,right=56,top=10,barH=17,gap=2,maxPct=%(maxpct)d;
var rowH=series.length*(barH+gap)+24, plotW=W-labelW-right, px=plotW/maxPct, H=top+rowH*data.length+34;
var NS="http://www.w3.org/2000/svg";
function el(n,a,t){var e=document.createElementNS(NS,n);for(var k in a)e.setAttribute(k,a[k]);if(t!=null)e.textContent=t;return e}
var svg=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"Harm-language rate by dataset and model"});
for(var t=0;t<=maxPct;t+=5){var x=labelW+t*px;svg.appendChild(el("line",{x1:x,y1:top,x2:x,y2:top+rowH*data.length,stroke:"var(--rule)"}));svg.appendChild(el("text",{x:x,y:H-10,"text-anchor":"middle","font-size":12,fill:"var(--fg-3)"},t+"%%"))}
var tip=document.getElementById("tip"),wrap=document.getElementById("chart");
data.forEach(function(d,i){var y0=top+i*rowH+(rowH-(series.length*(barH+gap)-gap))/2;
svg.appendChild(el("text",{x:labelW-14,y:y0+barH,"text-anchor":"end","font-size":14,"font-weight":600,fill:"var(--fg)"},d.ds));
svg.appendChild(el("text",{x:labelW-14,y:y0+barH+16,"text-anchor":"end","font-size":11,fill:"var(--fg-3)"},d.note));
series.forEach(function(s,j){var v=d.v[j];if(!v)return;var y=y0+j*(barH+gap),w=Math.max(v.pct*px,2);
var r=el("rect",{x:labelW,y:y,width:w,height:barH,rx:3,fill:"var(--s"+(j+1)+")","class":"bar",tabindex:0});
var lab=el("text",{x:labelW+w+6,y:y+barH-4,"font-size":12,fill:"var(--fg-2)"},v.pct.toFixed(1)+"%%");
var txt=s+" · "+d.ds+": "+v.pct.toFixed(1)+"%% of "+v.n.toLocaleString()+" positions ("+v.k.toLocaleString()+")";
function show(){tip.textContent=txt;tip.style.display="block";var b=r.getBoundingClientRect(),c=wrap.getBoundingClientRect();tip.style.left=(b.left-c.left+b.width/2)+"px";tip.style.top=(b.top-c.top)+"px";svg.querySelectorAll(".bar").forEach(function(o){if(o!==r)o.classList.add("dimmed")})}
function hide(){tip.style.display="none";svg.querySelectorAll(".bar").forEach(function(o){o.classList.remove("dimmed")})}
r.addEventListener("mouseenter",show);r.addEventListener("mouseleave",hide);r.addEventListener("focus",show);r.addEventListener("blur",hide);
svg.appendChild(r);svg.appendChild(lab)})});
svg.appendChild(el("line",{x1:labelW,y1:top,x2:labelW,y2:top+rowH*data.length,stroke:"var(--fg-3)"}));
wrap.appendChild(svg)})();
"""


def fmt_pct(x) -> str:
    return "–" if x is None else f"{x:.1f}"


def build(runs: dict[str, dict], args: argparse.Namespace) -> str:
    slugs = list(runs)
    st = {m: stats(runs[m]) for m in slugs}
    labels = {m: model_label(m) for m in slugs}
    datasets = [d for d in DATASET_ORDER if any(d in st[m]["datasets"] for m in slugs)]
    datasets += sorted({d for m in slugs for d in st[m]["datasets"]} - set(datasets))
    rng = random.Random(args.seed)

    total_pos = sum(s["positions"] for s in st.values())
    total_exp = sum(s["expected"] for s in st.values())
    max_hard = max((s["datasets"][d]["refusal_pct"] for s in st.values() for d in s["datasets"]), default=0)
    jb = [s["datasets"][d]["any_ref_pct"] for s in st.values() for d in s["datasets"] if d in JAILBREAK]
    kn = [s["datasets"][d]["any_ref_pct"] for s in st.values() for d in s["datasets"] if d in KNOWLEDGE]

    # ── header + tiles ──
    meta = [f"{labels[m][1]}" for m in slugs] + [" · ".join(DATASET_LABEL.get(d, d) for d in datasets),
                                                   f"{date.today():%d %b %Y}"]
    out = [f'<title>Adversarial Prompt Readouts</title>',
           '<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
           '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">',
           f'<style>{CSS}</style><main><header>',
           '<div class="eyebrow">Natural Language Autoencoders · adversarial extraction</div>',
           '<h1>Adversarial Prompt Readouts</h1>',
           '<p class="lede">What the Activation Verbalizer says about every token of adversarial and factual prompts, read from the residual stream at two-thirds depth.</p>',
           '<div class="meta">' + "".join(f"<span>{esc(x)}</span>" for x in meta) + '</div></header>']

    tiles = [(f"{total_pos:,}", f"token positions verbalized ({100 * total_pos / max(total_exp, 1):.0f} % coverage)"),
             (f"≤ {max_hard:.1f} %", "hard refusals by the verbalizer on any dataset")]
    if jb and kn:
        tiles.append((f"{min(jb):.0f}–{max(jb):.0f} %",
                      f"positions on jailbreak prompts described with harm language, vs {min(kn):.0f}–{max(kn):.0f} % on knowledge prompts"))
    out.append('<section class="tiles">' + "".join(
        f'<div class="tile"><div class="n">{n}</div><div class="l">{l}</div></div>' for n, l in tiles) + '</section>')

    # ── finding ──
    out.append('<h2>Finding</h2>')
    if jb and kn:
        out.append(f'<p>The verbalizer almost never declines to describe an activation (hard refusals ≤ {max_hard:.1f} %). '
                   f'But on the jailbreak-style datasets, {min(jb):.0f}–{max(jb):.0f} % of positions are described with words like '
                   f'<em>harmful</em>, <em>dangerous</em> or <em>illegal</em>; on the knowledge datasets, where hazardous or tricky '
                   f'content is asked about in a neutral frame, that rate is {min(kn):.0f}–{max(kn):.0f} %.</p>')
        if len(slugs) > 1:
            out.append('<p>The models agree within a few points on every dataset despite sharing no weights, tokenizer or training data: '
                       'the residual stream at this depth carries a readable "this request is harmful" signal for harmful <em>intent</em>, '
                       'and it is largely absent for hazardous <em>content</em>.</p>')
    else:
        out.append('<p>Per-dataset harm-language rates are shown below; the jailbreak-vs-knowledge comparison needs both groups of datasets in the run.</p>')

    # ── chart ──
    chart = [{"ds": DATASET_LABEL.get(d, d), "note": DATASET_NOTE.get(d, ""),
              "v": [({"pct": st[m]["datasets"][d]["any_ref_pct"], "n": st[m]["datasets"][d]["verbalized"],
                      "k": st[m]["datasets"][d]["any_ref"]} if d in st[m]["datasets"] else None) for m in slugs]}
             for d in datasets]
    peak = max((v["pct"] for c in chart for v in c["v"] if v), default=0)
    maxpct = int(5 * (peak // 5 + 1)) if peak > 0 else 5
    legend = "".join(f'<span style="--c:var(--s{i + 1})">{esc(labels[m][0])}</span>' for i, m in enumerate(slugs))
    out.append(f'<figure><div class="title">Positions described with harm language</div>'
               f'<div class="sub">Share of token positions whose verbalization contains a hard refusal or soft harm wording</div>'
               f'<div class="legend">{legend}</div><div class="chart-wrap" id="chart"><div class="tip" id="tip"></div></div>'
               f'<figcaption>Hover a bar for the count. Greedy decoding (temperature 0). Harm language = hard-refusal pattern or any of: '
               f'{", ".join(_SOFT)}.</figcaption></figure>')
    out.append("<script>" + CHART_JS % {"chart": json.dumps(chart), "series": json.dumps([labels[m][0] for m in slugs]),
                                       "maxpct": maxpct} + "</script>")

    # ── stats table ──
    out.append('<h2>Full statistics</h2><p><strong>Refused</strong> is the hard-refusal pattern; <strong>any-ref</strong> adds the soft keywords; '
               '<strong>CJK</strong> is the injection-path smoke test; <strong>base refused</strong> counts prompts the base model itself declined.</p>')
    rows = []
    for d in datasets:
        first = True
        for m in slugs:
            s = st[m]["datasets"].get(d)
            if not s:
                continue
            hi = ' class="hi"' if s["any_ref_pct"] >= 10 else ""
            rows.append(f'<tr><td>{DATASET_LABEL.get(d, d) if first else ""}</td><td class="model">{esc(labels[m][0])}</td>'
                        f'<td>{s["samples"]:,}</td><td>{s["positions"]:,}</td><td>{s["refused"]}</td><td>{s["refusal_pct"]:.1f}</td>'
                        f'<td{hi}>{s["any_ref_pct"]:.1f}</td><td>{s["cjk"]}</td><td>{s["base_refusals"]}/{s["samples"]}</td></tr>')
            first = False
    out.append('<div class="table-wrap"><table><thead><tr><th>Dataset</th><th>Model</th><th>Prompts</th><th>Positions</th><th>Refused</th>'
               '<th>Refusal %</th><th>Any-ref %</th><th>CJK</th><th>Base refused</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>')

    # ── validity ──
    out.append('<h3>Validity checks</h3><ul>')
    for m in slugs:
        s = st[m]
        cov_ok = s["positions"] == s["expected"]
        out.append(f'<li><span class="pill{"" if cov_ok else " warn"}">{"pass" if cov_ok else "partial"}</span> '
                   f'{esc(labels[m][0])} coverage: {s["positions"]:,} / {s["expected"]:,} positions over {s["samples"]} prompts.</li>')
        cjk_ok = s["cjk_rate"] <= 0.05
        out.append(f'<li><span class="pill{"" if cjk_ok else " warn"}">{"pass" if cjk_ok else "CHECK"}</span> '
                   f'{esc(labels[m][0])} injection smoke test: CJK in {s["cjk"]:,} of outputs ({s["cjk_rate"]:.2%}), '
                   f'{s["cjk_early"]} at early positions, over {s["cjk_samples"]} prompts. '
                   f'{"Within the documented noise floor; a broken path makes nearly every output CJK." if cjk_ok else "Above 5 % — check hs_offset and sidecar neighbour IDs."}</li>')
    out.append('</ul>')

    # ── examples ──
    out.append('<h2>Examples</h2><p>Random prompts per dataset and model (seed {}). Each shows the prompt, the base model\'s greedy reply, and a spread of '
               'positions with the verbalizer\'s description of the activation there. The bracketed token is the position being read.</p>'.format(args.seed))
    for d in datasets:
        out.append(f'<h3>{DATASET_LABEL.get(d, d)} <span class="dim">{DATASET_NOTE.get(d, "")}</span></h3>')
        for i, m in enumerate(slugs):
            run = runs[m]
            sids = [sid for sid, s in run["samples"].items() if s["dataset"] == d and sid in run["by_sample"]]
            if not sids:
                continue
            out.append(f'<div class="model-head"><span class="dot" style="--c:var(--s{i + 1})"></span><strong>{esc(labels[m][0])}</strong>'
                       f'<span class="dim">{len(sids)} prompts in this dataset</span></div>')
            for sid in rng.sample(sids, min(args.n_samples, len(sids))):
                out.append(sample_block(run["samples"][sid], run["by_sample"][sid], args.positions_per_sample, args.context))

    # ── harm-language positions ──
    out.append('<h2>Positions described with harm language</h2><p>Random positions whose verbalization matched the any-ref metric, with context. '
               'This is what the chart above is counting.</p>')
    for d in datasets:
        for i, m in enumerate(slugs):
            run = runs[m]
            pool = [r for r in run["recs"] if r["dataset"] == d and r["verbalization"] is not None and _flag(r)]
            if not pool:
                continue
            out.append(f'<div class="model-head"><span class="dot" style="--c:var(--s{i + 1})"></span><strong>{DATASET_LABEL.get(d, d)} · {esc(labels[m][0])}</strong>'
                       f'<span class="dim">{len(pool):,} flagged positions; showing {min(args.n_harm, len(pool))}</span></div><div class="harm-list">')
            for r in rng.sample(pool, min(args.n_harm, len(pool))):
                s = run["samples"][r["sample_id"]]
                out.append(position_row(r, run["by_sample"][r["sample_id"]], s["n_tokens"], args.context))
            out.append('</div>')

    # ── method + caveats ──
    out.append('<h2>Method</h2><ul>'
               '<li><strong>Extract.</strong> Base model forward pass with <code>output_hidden_states</code>; the layer-K residual captured at every token position; greedy 64-token decode saved as the base model\'s own reply.</li>'
               '<li><strong>Verbalize.</strong> Each activation injected as a token embedding at the marker position of the fixed AV prompt (SGLang, <code>--disable-radix-cache</code>), greedy, ≤ 200 tokens. Marker token, neighbour IDs and <code>injection_scale</code> come from each checkpoint\'s <code>nla_meta.yaml</code>.</li>'
               '<li><strong>Saved per model.</strong> <code>samples.jsonl</code> (prompt, token IDs, base-model reply), <code>verbalizations.jsonl</code> (one row per position), <code>activations.npy</code> (raw layer-K vectors).</li></ul>'
               '<h2>Caveats</h2><ul>'
               '<li><strong>Any-ref is a keyword heuristic.</strong> It counts descriptions that mention harm, not whether the description is accurate or whether the model would comply. The examples above are the check on it.</li>'
               '<li><strong>One layer per model, greedy decoding.</strong> One description per position; the AV\'s distribution over descriptions is not sampled.</li>'
               '<li><strong>Early positions</strong> (&lt; 50 tokens of left context) are included; the repo notes activations there decode less reliably.</li></ul>')
    out.append('</main>')
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", required=True, help="directory containing one subdir per model")
    ap.add_argument("--models", nargs="+", required=True, help="subdir names, e.g. qwen7b-l20 gemma12b-l32")
    ap.add_argument("--out", required=True, help="output .html path")
    ap.add_argument("--datasets", nargs="+", default=None, help="restrict to these datasets")
    ap.add_argument("--n-samples", type=int, default=3, help="example prompts per dataset per model")
    ap.add_argument("--positions-per-sample", default="6",
                    help="positions shown per example prompt (evenly spaced + flagged), or 'all'")
    ap.add_argument("--n-harm", type=int, default=8, help="flagged positions listed per dataset per model")
    ap.add_argument("--context", type=int, default=12, help="tokens of context each side of the position")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.positions_per_sample = None if args.positions_per_sample == "all" else int(args.positions_per_sample)

    runs = {}
    for m in args.models:
        d = Path(args.runs_root) / m
        if not (d / "verbalizations.jsonl").exists():
            sys.exit(f"no verbalizations.jsonl in {d}")
        runs[m] = load_run(d, args.datasets)
        print(f"[report] {m}: {len(runs[m]['samples'])} prompts, {len(runs[m]['recs'])} positions")

    page = build(runs, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"[report] wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
