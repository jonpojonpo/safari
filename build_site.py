#!/usr/bin/env python3
"""Build the Safari Field Journal static site (field-guide / dossier design).

Reads the existing classification + editorial JSON and the generated image
assets, then emits a single self-contained index.html. No network or cloud
API required — purely a presentation step over data already on disk.
"""
from __future__ import annotations

import html
import json
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Merge inconsistent labels into a single canonical animal.
CANON = {
    "Rhino": "White Rhino",
    "Kingfisher": "Malachite Kingfisher",
    "Bulbul": "Dark-capped Bulbul",
    "Chital": "Impala",
    "Tiger": "Unknown",
}
EXCLUDE = {"No Animal", "Unknown", "None", "Bird"}


def esc(value: str) -> str:
    return html.escape(str(value), quote=True)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def stars(score: float) -> int:
    """Normalise a quality score to a 1–5 star rating."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0
    if s > 5:  # 0–10 scale
        s = s / 2
    return max(0, min(5, round(s)))


def img_or(*candidates: str) -> str:
    for c in candidates:
        if (ROOT / c).exists():
            return c
    return candidates[-1]


def load():
    classifications = json.loads((DATA / "classifications.json").read_text())
    info = json.loads((DATA / "animal_info.json").read_text())
    return classifications, info


def group_animals(classifications: dict):
    groups: dict[str, list[dict]] = defaultdict(list)
    for fn, v in classifications.items():
        animal = CANON.get(v.get("animal", ""), v.get("animal", ""))
        if animal in EXCLUDE:
            continue
        stem = os.path.splitext(v.get("file", fn))[0]
        if not (DATA / "web" / f"{stem}.jpg").exists():
            continue
        rec = dict(v)
        rec["stem"] = stem
        rec["animal"] = animal
        groups[animal].append(rec)
    for photos in groups.values():
        photos.sort(key=lambda r: (r.get("quality_score", 0), r.get("technical_score", 0)), reverse=True)
    # Order animals by how many strong photos we have, then name.
    return dict(sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def render_gallery(photos: list[dict]) -> str:
    items = []
    for r in photos:
        stem = r["stem"]
        thumb = img_or(f"data/thumbs/{stem}_thumb.jpg", f"data/web/{stem}.jpg")
        full = f"data/web/{stem}.jpg"
        behaviour = esc(r.get("behaviour", "") or "")
        notes = esc(r.get("notes", "") or "")
        st = "★" * stars(r.get("quality_score", 0))
        items.append(f"""
          <button class="shot" data-full="{full}" data-behaviour="{behaviour}" data-notes="{notes}" data-stars="{st}">
            <img src="{thumb}" alt="{behaviour}" loading="lazy">
            <span class="shot-meta"><span class="shot-stars">{st}</span></span>
          </button>""")
    return "".join(items)


def render_facts(facts) -> str:
    if not isinstance(facts, list):
        return ""
    lis = "".join(f"<li>{esc(f)}</li>" for f in facts)
    return f'<ul class="facts">{lis}</ul>'


def block(label: str, text: str) -> str:
    if not text:
        return ""
    return f"""
        <section class="field">
          <h3>{esc(label)}</h3>
          <p>{esc(text)}</p>
        </section>"""


def render_dossier(animal: str, photos: list[dict], info: dict) -> str:
    meta = info.get(animal, {})
    best = photos[0]
    sci = best.get("scientific_name", "") or ""
    hero = f"data/web/{best['stem']}.jpg"
    best_stars = "★" * stars(best.get("quality_score", 0))
    top_q = max((stars(p.get("quality_score", 0)) for p in photos), default=0)
    top_t = max((stars(p.get("technical_score", 0)) for p in photos), default=0)
    sightings = sum(int(p.get("animal_count", 1) or 1) for p in photos)
    tagline = meta.get("tagline", "")
    intro = meta.get("intro", "")

    return f"""
      <section class="dossier" id="{slug(animal)}">
        <figure class="hero">
          <img src="{hero}" alt="{esc(animal)}">
          <figcaption>
            <span class="hero-sci">{esc(sci)}</span>
            <h2 class="hero-name">{esc(animal)}</h2>
            <span class="hero-best">Best shot {best_stars}</span>
          </figcaption>
        </figure>

        {f'<p class="tagline">{esc(tagline)}</p>' if tagline else ''}
        {f'<p class="lead">{esc(intro)}</p>' if intro else ''}

        {render_facts(meta.get("fast_facts"))}

        {block("Habitat & Range", meta.get("habitat", ""))}
        {block("Behaviour", meta.get("behaviour", ""))}
        {block("Conservation", meta.get("conservation", ""))}

        <section class="encounter">
          <h3>Field Encounter</h3>
          <div class="enc-grid">
            <div><span class="enc-num">{len(photos)}</span><span class="enc-lbl">Photographs</span></div>
            <div><span class="enc-num">{sightings}</span><span class="enc-lbl">Individuals seen</span></div>
            <div><span class="enc-num">{top_q}/5</span><span class="enc-lbl">Top quality</span></div>
            <div><span class="enc-num">{top_t}/5</span><span class="enc-lbl">Top technical</span></div>
          </div>
        </section>

        <section class="field">
          <h3>Gallery</h3>
          <div class="gallery">{render_gallery(photos)}</div>
        </section>
      </section>"""


def render_nav(groups: dict) -> str:
    rows = []
    for animal, photos in groups.items():
        rows.append(
            f'<li><a href="#{slug(animal)}" class="nav-link">'
            f'<span>{esc(animal)}</span><span class="nav-count">{len(photos)}</span></a></li>'
        )
    return "".join(rows)


def build(groups: dict, info: dict, analysed: int) -> str:
    species = len(groups)
    photos = sum(len(p) for p in groups.values())
    sections = "".join(render_dossier(a, p, info) for a, p in groups.items())
    nav = render_nav(groups)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Safari Field Journal</title>
<meta name="description" content="An African wildlife field journal — {species} species, {photos} photographs, AI-curated from {analysed} frames.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,400&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
{CSS}
</style>
</head>
<body>
<button class="menu-btn" id="menuBtn" aria-label="Toggle index">☰ Index</button>
<div class="scrim" id="scrim"></div>

<aside class="sidebar" id="sidebar">
  <a class="brand" href="#top">
    <span class="brand-mark">Safari</span>
    <span class="brand-sub">Field Journal</span>
  </a>
  <p class="brand-stat">{species} species · {photos} photographs</p>
  <nav><ul class="nav-list">{nav}</ul></nav>
  <p class="sidebar-foot">Curated with computer vision from {analysed} frames.</p>
</aside>

<main class="content" id="top">
  <header class="masthead">
    <span class="eyebrow">A Wildlife Field Journal</span>
    <h1>Safari</h1>
    <p class="masthead-sub">An intimate field record of Africa's wild creatures — every frame scored, sorted, and captioned by machine vision, then bound into a single illustrated journal.</p>
    <div class="masthead-stats">
      <div><span class="ms-num">{species}</span><span class="ms-lbl">Species</span></div>
      <div><span class="ms-num">{photos}</span><span class="ms-lbl">Photographs</span></div>
      <div><span class="ms-num">{analysed}</span><span class="ms-lbl">Frames analysed</span></div>
    </div>
  </header>
  {sections}
  <footer class="site-foot">
    <div class="foot-rule"></div>
    <p class="foot-title">Safari Field Journal</p>
    <p class="foot-sub">{species} species · {photos} photographs · {analysed} frames analysed</p>
  </footer>
</main>

<div class="lightbox" id="lightbox" onclick="if(event.target===this)closeLb()">
  <button class="lb-close" onclick="closeLb()" aria-label="Close">✕</button>
  <img class="lb-img" id="lbImg" alt="">
  <div class="lb-cap">
    <p class="lb-stars" id="lbStars"></p>
    <p class="lb-behaviour" id="lbBehaviour"></p>
    <p class="lb-notes" id="lbNotes"></p>
  </div>
</div>

<script>
{JS}
</script>
</body>
</html>"""


CSS = r"""
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d0d0f; --bg2:#141417; --panel:#17171b; --line:#26262c;
  --gold:#cba135; --gold-soft:#e6c66a; --ink:#e7e2d6; --dim:#938b7c; --faint:#5c554a;
  --serif:'Fraunces',Georgia,serif; --sans:'Inter',system-ui,sans-serif;
  --side:268px;
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:17px;line-height:1.7;-webkit-font-smoothing:antialiased;overflow-x:hidden}
p,h1,h2,h3,li,.tagline,.lead{overflow-wrap:break-word}
img{display:block;max-width:100%}

/* ── Sidebar ── */
.sidebar{position:fixed;top:0;left:0;width:var(--side);height:100vh;overflow-y:auto;
  background:linear-gradient(180deg,#101013,#0a0a0c);border-right:1px solid var(--line);
  padding:2rem 1.4rem;z-index:60}
.brand{display:block;text-decoration:none;margin-bottom:.4rem}
.brand-mark{display:block;font-family:var(--serif);font-weight:700;font-size:2rem;color:#fff;line-height:1}
.brand-sub{display:block;font-size:.66rem;letter-spacing:.34em;text-transform:uppercase;color:var(--gold);margin-top:.45rem}
.brand-stat{font-size:.72rem;color:var(--dim);margin:1rem 0 1.4rem;padding-bottom:1.2rem;border-bottom:1px solid var(--line)}
.nav-list{list-style:none}
.nav-link{display:flex;justify-content:space-between;align-items:center;gap:.6rem;
  padding:.4rem .6rem;border-radius:6px;text-decoration:none;color:var(--dim);
  font-size:.82rem;border-left:2px solid transparent;transition:.18s}
.nav-link span:first-child{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.nav-count{font-size:.62rem;color:var(--faint);background:var(--panel);border:1px solid var(--line);
  border-radius:10px;padding:.05rem .4rem;flex-shrink:0}
.nav-link:hover{color:var(--ink);background:rgba(203,161,53,.06)}
.nav-link.active{color:var(--gold-soft);border-left-color:var(--gold);background:rgba(203,161,53,.08)}
.nav-link.active .nav-count{color:var(--gold);border-color:var(--gold)}
.sidebar-foot{margin-top:1.6rem;padding-top:1.2rem;border-top:1px solid var(--line);font-size:.68rem;color:var(--faint);line-height:1.5}

/* ── Content ── */
.content{margin-left:var(--side);max-width:880px;padding:0 clamp(1.4rem,5vw,4.5rem) 4rem}
.masthead{min-height:min(86vh,680px);display:flex;flex-direction:column;justify-content:center;
  border-bottom:1px solid var(--line);padding:5rem 0 4rem}
.eyebrow{font-size:.72rem;letter-spacing:.32em;text-transform:uppercase;color:var(--gold)}
.masthead h1{font-family:var(--serif);font-weight:700;font-size:clamp(3.5rem,11vw,7rem);
  color:#fff;line-height:.95;margin:1.2rem 0}
.masthead-sub{font-family:var(--serif);font-size:clamp(1.1rem,2.2vw,1.5rem);font-style:italic;
  color:var(--dim);max-width:34ch;line-height:1.5}
.masthead-stats{display:flex;gap:3rem;flex-wrap:wrap;margin-top:3rem}
.ms-num{display:block;font-family:var(--serif);font-weight:700;font-size:2.6rem;color:var(--gold);line-height:1}
.ms-lbl{display:block;font-size:.66rem;letter-spacing:.18em;text-transform:uppercase;color:var(--faint);margin-top:.5rem}

/* ── Dossier ── */
.dossier{padding:5rem 0;border-bottom:1px solid var(--line);scroll-margin-top:1.5rem}
.hero{position:relative;border-radius:10px;overflow:hidden;aspect-ratio:16/10;background:#000}
.hero img{width:100%;height:100%;object-fit:cover}
.hero figcaption{position:absolute;inset:auto 0 0 0;padding:2.4rem 2rem 1.6rem;
  background:linear-gradient(transparent,rgba(0,0,0,.55) 40%,rgba(0,0,0,.9))}
.hero-sci{font-family:var(--serif);font-style:italic;color:var(--gold-soft);font-size:1rem}
.hero-name{font-family:var(--serif);font-weight:700;color:#fff;font-size:clamp(2rem,5vw,3.4rem);line-height:1.05;margin:.2rem 0 .5rem}
.hero-best{font-size:.66rem;letter-spacing:.16em;text-transform:uppercase;color:var(--dim)}
.tagline{font-family:var(--serif);font-style:italic;font-size:1.35rem;color:var(--gold-soft);margin:1.8rem 0 1rem;line-height:1.4}
.lead{font-family:var(--serif);font-size:1.2rem;line-height:1.7;color:#cfc8ba;margin-bottom:1.5rem}
.facts{list-style:none;display:grid;grid-template-columns:1fr;gap:0;margin:1.5rem 0 .5rem;
  border-top:1px solid var(--line)}
.facts li{font-size:.92rem;color:var(--ink);padding:.7rem 0 .7rem 1.4rem;border-bottom:1px solid var(--line);position:relative}
.facts li::before{content:'';position:absolute;left:0;top:1.15rem;width:6px;height:6px;border-radius:50%;background:var(--gold)}
.field{margin-top:2.4rem}
.field h3,.encounter h3{font-family:var(--sans);font-weight:600;font-size:.7rem;letter-spacing:.22em;
  text-transform:uppercase;color:var(--gold);padding-bottom:.5rem;border-bottom:1px solid var(--line);margin-bottom:1rem}
.field p{font-family:var(--serif);font-size:1.08rem;color:#c8c1b4}

/* ── Encounter ── */
.encounter{margin-top:2.6rem}
.enc-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);
  border:1px solid var(--line);border-radius:8px;overflow:hidden}
.enc-grid div{background:var(--bg2);padding:1.1rem .8rem;text-align:center}
.enc-num{display:block;font-family:var(--serif);font-weight:700;font-size:1.7rem;color:var(--gold-soft)}
.enc-lbl{display:block;font-size:.6rem;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);margin-top:.35rem}

/* ── Gallery ── */
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px}
.shot{position:relative;border:0;padding:0;cursor:pointer;background:#000;border-radius:6px;
  overflow:hidden;aspect-ratio:3/2}
.shot img{width:100%;height:100%;object-fit:cover;transition:transform .4s ease,opacity .3s}
.shot:hover img{transform:scale(1.06)}
.shot-meta{position:absolute;inset:auto 0 0 0;padding:.7rem .6rem .4rem;
  background:linear-gradient(transparent,rgba(0,0,0,.8));opacity:0;transition:.25s;text-align:left}
.shot:hover .shot-meta{opacity:1}
.shot-stars{color:var(--gold);font-size:.78rem;letter-spacing:.05em}

/* ── Footer ── */
.site-foot{text-align:center;padding:4rem 0 1rem}
.foot-rule{width:54px;height:2px;background:var(--gold);margin:0 auto 1.4rem}
.foot-title{font-family:var(--serif);font-size:1.25rem;color:var(--dim)}
.foot-sub{font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);margin-top:.5rem}

/* ── Lightbox ── */
.lightbox{display:none;position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.96);
  flex-direction:column;align-items:center;justify-content:center;padding:2rem}
.lightbox.open{display:flex}
.lb-img{max-width:92vw;max-height:78vh;object-fit:contain;border-radius:6px}
.lb-cap{text-align:center;max-width:640px;margin-top:1.2rem}
.lb-stars{color:var(--gold);letter-spacing:.1em;font-size:1rem}
.lb-behaviour{font-family:var(--serif);font-style:italic;font-size:1.2rem;color:var(--ink);margin-top:.3rem}
.lb-notes{font-size:.86rem;color:var(--dim);margin-top:.5rem;line-height:1.6}
.lb-close{position:fixed;top:1.3rem;right:1.3rem;width:44px;height:44px;border-radius:50%;
  background:none;border:1px solid #444;color:var(--ink);font-size:1.1rem;cursor:pointer;transition:.2s}
.lb-close:hover{border-color:var(--gold);color:var(--gold)}

/* ── Mobile ── */
.menu-btn,.scrim{display:none}
@media(max-width:960px){
  :root{--side:0px}
  .sidebar{transform:translateX(-100%);transition:transform .28s ease;width:280px}
  .sidebar.open{transform:none;box-shadow:0 0 50px rgba(0,0,0,.7)}
  .content{margin-left:0;padding-top:3.5rem}
  .menu-btn{display:block;position:fixed;top:1rem;left:1rem;z-index:80;
    background:var(--panel);border:1px solid var(--line);color:var(--gold-soft);
    font-size:.8rem;letter-spacing:.1em;padding:.55rem .9rem;border-radius:8px;cursor:pointer}
  .scrim{position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:55}
  .scrim.open{display:block}
  .enc-grid{grid-template-columns:repeat(2,1fr)}
}
@media(max-width:480px){
  .gallery{grid-template-columns:repeat(2,1fr)}
}
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:#2c2c33;border-radius:5px}
::-webkit-scrollbar-thumb:hover{background:var(--gold)}
"""

JS = r"""
const sidebar=document.getElementById('sidebar');
const scrim=document.getElementById('scrim');
const menuBtn=document.getElementById('menuBtn');
function toggleNav(open){sidebar.classList.toggle('open',open);scrim.classList.toggle('open',open);}
menuBtn.addEventListener('click',()=>toggleNav(!sidebar.classList.contains('open')));
scrim.addEventListener('click',()=>toggleNav(false));
document.querySelectorAll('.nav-link').forEach(a=>a.addEventListener('click',()=>toggleNav(false)));

// Scroll-spy
const links=new Map();
document.querySelectorAll('.nav-link').forEach(a=>links.set(a.getAttribute('href').slice(1),a));
const spy=new IntersectionObserver((entries)=>{
  entries.forEach(e=>{
    if(e.isIntersecting){
      links.forEach(l=>l.classList.remove('active'));
      const l=links.get(e.target.id);
      if(l){l.classList.add('active');l.scrollIntoView({block:'nearest'});}
    }
  });
},{rootMargin:'-45% 0px -50% 0px'});
document.querySelectorAll('.dossier').forEach(s=>spy.observe(s));

// Lightbox
const lb=document.getElementById('lightbox');
function openShot(d){
  document.getElementById('lbImg').src=d.full;
  document.getElementById('lbStars').textContent=d.stars||'';
  document.getElementById('lbBehaviour').textContent=d.behaviour||'';
  document.getElementById('lbNotes').textContent=d.notes||'';
  lb.classList.add('open');document.body.style.overflow='hidden';
}
document.querySelectorAll('.shot').forEach(b=>b.addEventListener('click',()=>openShot(b.dataset)));
function closeLb(){lb.classList.remove('open');document.body.style.overflow='';}
window.closeLb=closeLb;
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeLb();});
"""


def main() -> int:
    classifications, info = load()
    groups = group_animals(classifications)
    analysed = len(classifications)
    html_out = build(groups, info, analysed)
    (ROOT / "index.html").write_text(html_out, encoding="utf-8")
    print(f"Built index.html — {len(groups)} species, "
          f"{sum(len(p) for p in groups.values())} photos, {analysed} frames analysed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
