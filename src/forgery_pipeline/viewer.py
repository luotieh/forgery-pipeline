"""数据集可视化：生成自包含 viewer.html（静态 HTML 生成器）。"""
from __future__ import annotations
import json
import os
from pathlib import Path
import cv2
import numpy as np
from forgery_pipeline import image_io, manifest

_HTML_TEMPLATE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>Forgery Dataset Viewer</title>
<style>
body{margin:0;font-family:system-ui,Arial,sans-serif;background:#0f1115;color:#e6e6e6}
header{padding:10px 14px;background:#171a21;position:sticky;top:0;border-bottom:1px solid #2a2f3a}
header h1{font-size:15px;margin:0 0 8px}
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
select,input{background:#0f1115;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:6px;padding:4px 6px;font-size:12px}
.count{font-size:12px;color:#9aa4b2;margin-left:auto}
main{display:flex;height:calc(100vh - 92px)}
.gallery{width:46%;overflow:auto;padding:8px;display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:8px;align-content:start}
.card{cursor:pointer;border:2px solid transparent;border-radius:8px;overflow:hidden;background:#171a21}
.card.fake{border-color:#7a2230}.card.real{border-color:#234e2a}.card.sel{outline:2px solid #4f8cff}
.card img{width:100%;display:block;aspect-ratio:1/1;object-fit:cover}
.card .cap{font-size:10px;padding:2px 4px;color:#9aa4b2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.detail{flex:1;overflow:auto;padding:14px;border-left:1px solid #2a2f3a}
.imgs{display:flex;gap:10px;flex-wrap:wrap}.imgs figure{margin:0}.imgs figcaption{font-size:11px;color:#9aa4b2;text-align:center}
.imgs img{max-width:240px;max-height:240px;border:1px solid #2a2f3a;border-radius:6px}
table{border-collapse:collapse;margin-top:12px;font-size:12px}td{border:1px solid #2a2f3a;padding:3px 8px}.k{color:#9aa4b2}
.expl{margin-top:12px;font-size:12px;line-height:1.5;background:#171a21;padding:10px;border-radius:8px}
.badge{display:inline-block;font-size:11px;padding:1px 6px;border-radius:10px;margin-right:4px}.b-fake{background:#7a2230}.b-real{background:#234e2a}
</style></head><body>
<header><h1>Forgery Dataset Viewer</h1><div class="filters" id="filters"></div></header>
<main><div class="gallery" id="gallery"></div>
<div class="detail" id="detail"><p style="color:#9aa4b2">点击左侧缩略图查看详情</p></div></main>
<script>
const SAMPLES = __SAMPLES__;
const FIELDS=[["split","split"],["task_type","task"],["manipulation_level1","level1"],["manipulation_level3","level3"],["generator_family","gen"]];
const state={};
function esc(x){return String(x==null?'':x).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function distinct(f){return [...new Set(SAMPLES.map(s=>s[f]).filter(v=>v!=null))].sort();}
function buildFilters(){
 const c=document.getElementById('filters');
 for(const [f,label] of FIELDS){const sel=document.createElement('select');
  sel.innerHTML=`<option value="">${label}: 全部</option>`+distinct(f).map(v=>`<option>${esc(v)}</option>`).join('');
  sel.onchange=()=>{state[f]=sel.value;render();};c.appendChild(sel);}
 const fk=document.createElement('select');
 fk.innerHTML='<option value="">真/假: 全部</option><option value="1">fake</option><option value="0">real</option>';
 fk.onchange=()=>{state.is_fake=fk.value;render();};c.appendChild(fk);
 const q=document.createElement('input');q.placeholder='搜索 image_id';q.oninput=()=>{state.q=q.value;render();};c.appendChild(q);
 const cnt=document.createElement('span');cnt.className='count';cnt.id='count';c.appendChild(cnt);}
function match(s){
 for(const [f] of FIELDS){if(state[f]&&String(s[f])!==state[f])return false;}
 if(state.is_fake!==undefined&&state.is_fake!==''&&String(s.is_fake)!==state.is_fake)return false;
 if(state.q&&!s.image_id.includes(state.q))return false;return true;}
function render(){
 const g=document.getElementById('gallery');const list=SAMPLES.filter(match);
 document.getElementById('count').textContent=`${list.length} / ${SAMPLES.length} 样本`;g.innerHTML='';
 for(const s of list){const d=document.createElement('div');d.className='card '+(s.is_fake?'fake':'real');
  d.innerHTML=`<img loading="lazy" src="${s.paths.thumb}"><div class="cap">${esc(s.image_id)}</div>`;
  d.onclick=()=>{showDetail(s);[...g.children].forEach(x=>x.classList.remove('sel'));d.classList.add('sel');};
  g.appendChild(d);}}
function fig(src,cap){return src?`<figure><img src="${src}"><figcaption>${cap}</figcaption></figure>`:'';}
function row(k,v){return v==null||v===''?'':`<tr><td class="k">${k}</td><td>${esc(v)}</td></tr>`;}
function showDetail(s){const e=s.explanation;
 const imgs=fig(s.paths.real,'原图 real')+fig(s.paths.fake,'伪造 fake')+fig(s.paths.overlay,'mask 叠加');
 const expl=e?`<div class="expl"><b>解释</b><br>位置: ${esc(e.location_description)}<br>伪迹: ${esc(e.visual_artifact_description)}<br>推理: ${esc(e.semantic_reasoning)}<br>结论: ${esc(e.forensic_conclusion)}</div>`:'';
 document.getElementById('detail').innerHTML=
  `<div><span class="badge ${s.is_fake?'b-fake':'b-real'}">${s.is_fake?'FAKE':'REAL'}</span><b>${esc(s.image_id)}</b></div>`+
  `<div class="imgs">${imgs||'<p style=\\'color:#9aa4b2\\'>无图像</p>'}</div>`+
  `<table>${row('task_type',s.task_type)}${row('split',s.split)}${row('level1',s.manipulation_level1)}${row('level3',s.manipulation_level3)}${row('generator',s.generator_name)}${row('family',s.generator_family)}${row('quality_score',s.quality_score)}${row('prompt',s.prompt)}</table>`+expl;}
buildFilters();render();
</script></body></html>
"""


def render_overlay(fake: np.ndarray, mask: np.ndarray,
                   color: tuple = (255, 0, 0), alpha: float = 0.35) -> np.ndarray:
    """在伪造图上叠加 mask：mask 区域半透明红色填充 + 红色轮廓。"""
    out = fake.astype(np.float32).copy()
    m = mask > 127
    col = np.array(color, np.float32)
    out[m] = (1.0 - alpha) * out[m] + alpha * col
    out = out.astype(np.uint8)
    contours, _ = cv2.findContours(m.astype(np.uint8) * 255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def make_thumb(img: np.ndarray, size: int = 224) -> np.ndarray:
    """等比缩放使长边=size（不放大）。"""
    h, w = img.shape[:2]
    scale = min(1.0, size / max(h, w))
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _rel(path_abs: Path, start_dir: Path) -> str:
    return os.path.relpath(path_abs, start_dir).replace(os.sep, "/")


def build_viewer(run_dir, out_html=None, max_samples=None) -> Path:
    """读 <run>/manifest.jsonl，生成 thumb/overlay 与自包含 viewer.html。"""
    run_dir = Path(run_dir)
    samples = manifest.read_jsonl(run_dir / "manifest.jsonl")
    if max_samples:
        samples = samples[:max_samples]
    out_html = Path(out_html) if out_html else run_dir / "viewer.html"
    out_dir = out_html.parent
    thumb_dir = run_dir / "viewer_assets" / "thumb"
    overlay_dir = run_dir / "viewer_assets" / "overlay"

    records, skipped = [], 0
    for s in samples:
        fake_abs = run_dir / s.image_path
        try:
            fimg = image_io.load_image(fake_abs)
        except Exception:
            skipped += 1
            continue
        thumb_abs = thumb_dir / f"{s.image_id}.jpg"
        image_io.save_image(make_thumb(fimg), thumb_abs)

        overlay_rel = None
        if s.mask_path and (run_dir / s.mask_path).exists():
            try:
                ov = render_overlay(fimg, image_io.load_mask(run_dir / s.mask_path))
                overlay_abs = overlay_dir / f"{s.image_id}.jpg"
                image_io.save_image(ov, overlay_abs)
                overlay_rel = _rel(overlay_abs, out_dir)
            except Exception:
                overlay_rel = None

        real_rel = None
        if s.real_image_path and (run_dir / s.real_image_path).exists():
            real_rel = _rel(run_dir / s.real_image_path, out_dir)

        records.append({
            "image_id": s.image_id, "is_fake": s.is_fake,
            "task_type": s.task_type.value, "split": s.split,
            "manipulation_level1": s.manipulation_level1,
            "manipulation_level3": s.manipulation_level3,
            "generator_family": s.generator_family,
            "generator_name": s.generator_name,
            "quality_score": s.quality_score, "prompt": s.prompt,
            "explanation": s.explanation.model_dump() if s.explanation else None,
            "paths": {"real": real_rel, "fake": _rel(fake_abs, out_dir),
                      "overlay": overlay_rel, "thumb": _rel(thumb_abs, out_dir)},
        })

    payload = json.dumps(records, ensure_ascii=False).replace("<", "\\u003c")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(_HTML_TEMPLATE.replace("__SAMPLES__", payload), encoding="utf-8")
    if skipped:
        print(f"viewer: 跳过 {skipped} 个无法读取的样本")
    return out_html
