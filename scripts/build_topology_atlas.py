"""Build a self-contained Russian HTML atlas from Markdown and rendered Mermaid SVGs.

Documentation-only dependency: markdown-it-py (tables enabled). Render each source
with @mermaid-js/mermaid-cli 12.2.0, using output stems airllm, colibri, overview
and artefacts directory docs/assets/topology. No runtime model dependencies.
"""
import base64
import hashlib
import html
import json
import re
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGES = [
    ("overview", "Общая карта", "ORIGINALS_TOPOLOGY_RU.md"),
    ("airllm", "AirLLM отдельно", "AIRLLM_ORIGINAL_TOPOLOGY_RU.md"),
    ("colibri", "Colibrì отдельно", "COLIBRI_ORIGINAL_TOPOLOGY_RU.md"),
    ("overlap", "Наложение функций", "FUNCTIONAL_OVERLAY_RU.md"),
    ("plan", "План объединения", "INTEGRATION_INTERSECTIONS_PLAN_RU.md"),
    ("graphify", "Graphify", "GRAPHIFY_AUDIT_RU.md"),
    ("sources", "Исходники и границы", "ORIGINALS_SOURCE_INDEX.md"),
    ("compact", "Последние испытания", "COMPACT_MOE_RESULTS_2026-11-13.md"),
]

STYLE = """
:root{color-scheme:light;--ink:#172f3a;--muted:#52636d;--paper:#f5f4ef;--green:#176c58;--line:#d9e2de}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.65 'Segoe UI',Arial,sans-serif}
header{background:#102f39;color:#f6faf7;padding:36px max(24px,calc((100vw - 1160px)/2));border-bottom:6px solid #7dc9ae}
header h1{font-size:clamp(29px,4vw,48px);line-height:1.12;margin:12px 0 18px;letter-spacing:-1px}header p{max-width:850px;color:#cededb;margin:8px 0}.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#b9ddcf}
nav{display:flex;flex-wrap:wrap;gap:8px;position:sticky;top:0;z-index:3;background:#f5f4eff5;padding:14px max(20px,calc((100vw - 1160px)/2));border-bottom:1px solid var(--line)}
nav a{font-size:14px;text-decoration:none;border:1px solid #becdc5;background:#fff;border-radius:6px;padding:8px 13px;color:var(--ink)}nav a[aria-current=page]{background:var(--green);border-color:var(--green);color:white}
main{max-width:1208px;margin:0 auto;padding:26px 24px 80px}article{background:#fff;padding:28px 36px 44px;border:1px solid var(--line);border-radius:10px;box-shadow:0 8px 24px #173b3010}article[hidden]{display:none}
article h1{font-size:29px;line-height:1.25;margin:4px 0 24px}h2{font-size:23px;line-height:1.35;margin:44px 0 16px;padding-top:20px;border-top:1px solid var(--line)}h3{font-size:19px;margin-top:30px}p{margin:14px 0}a{color:#155f96;text-underline-offset:3px}code{font:14px/1.5 Consolas,monospace;overflow-wrap:anywhere;background:#edf3f0;padding:2px 4px;border-radius:3px}pre{overflow:auto;background:#edf3f0;padding:20px}
.table-wrap{overflow-x:auto;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:15px}th{text-align:left;background:#e9f2ee;color:#214c42}td,th{padding:12px 14px;vertical-align:top;border:1px solid #d9e3dd}tr:nth-child(even) td{background:#fafbf9}
figure{margin:25px 0;padding:18px;background:#fbfcfa;border:1px solid #dce5df;border-radius:8px;overflow:auto}figure button{display:block;width:100%;padding:0;border:0;background:transparent;cursor:zoom-in}figure img{display:block;max-width:100%;height:auto;margin:auto}figcaption{font-size:12px;color:var(--muted);margin-top:14px;text-align:center}
li{margin:12px 0}p,li,a{overflow-wrap:anywhere}blockquote{border-left:4px solid #cfad57;margin:20px 0;padding:5px 18px;background:#fff8e8}.source-ref{font-size:13px;white-space:nowrap}footer{max-width:1160px;margin:0 auto 30px;color:var(--muted);font-size:13px;padding:0 24px}
.toc{background:#f1f6f3;border:1px solid #d4e0d8;padding:12px 18px;border-radius:6px;margin:18px 0}.toc summary{cursor:pointer;font-weight:600}.toc li{margin:5px 0}h2[id],h3[id]{scroll-margin-top:100px}
dialog{width:96vw;max-width:1800px;height:94vh;padding:14px;border:1px solid #99ada5;border-radius:8px}dialog::backdrop{background:#10252cbd}.zoom-area{height:calc(100% - 45px);overflow:auto;background:white}dialog img{max-width:none;display:block;margin:auto}#close-zoom{position:sticky;top:0;background:#173b43;color:white;border:0;border-radius:5px;padding:10px 18px;cursor:pointer;margin-bottom:8px}
@media(max-width:700px){body{font-size:16px}header{padding:25px 18px}main{padding:14px 10px 45px}article{padding:20px 16px}nav{padding:9px 10px;position:relative}nav a{font-size:12px;padding:7px 9px}figure{padding:8px}h2{font-size:21px}td,th{padding:9px}}
@media print{nav,header,footer,dialog{display:none}article[hidden]{display:block}main{max-width:none;padding:0}article{border:0;box-shadow:none;page-break-before:always}figure,table{break-inside:avoid}figure img{max-height:240mm}}
"""


def build():
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    originals = md.renderer.rules["fence"]
    counter = 0
    stem = ""
    diagrams = []

    def fence(tokens, idx, options, env):
        nonlocal counter
        token = tokens[idx]
        if token.info.strip() != "mermaid":
            return originals(tokens, idx, options, env)
        counter += 1
        path = DOCS / "assets/topology" / f"{stem}-{counter}.svg"
        raw = path.read_bytes()
        if b"<svg" not in raw or b"Syntax error" in raw:
            raise ValueError(f"Invalid rendered diagram: {path}")
        diagrams.append({"file": str(path.relative_to(ROOT)).replace('\\', '/'),
                         "sha256": hashlib.sha256(raw).hexdigest()})
        uri = "data:image/svg+xml;base64," + base64.b64encode(raw).decode()
        label = html.escape(f"{stem}: схема {counter}. Нажать для увеличения")
        return f'<figure><button type="button" class="zoom" aria-label="{label}"><img alt="{label}" src="{uri}"></button><figcaption>Схема {counter} · нажмите для увеличения · исходный Mermaid сохранён в Markdown</figcaption></figure>\n'

    md.renderer.rules["fence"] = fence
    articles = []
    for key, title, filename in PAGES:
        stem, counter = key, 0
        text = (DOCS / filename).read_text(encoding="utf-8")
        body = md.render(text)
        body = re.sub(r"<table>(.*?)</table>", r'<div class="table-wrap"><table>\1</table></div>', body, flags=re.S)
        for dest, _, source in PAGES:
            body = body.replace(f'href="{source}"', f'href="#{dest}"')
        body = body.replace('href="ORIGINALS_TOPOLOGY_RU.html"', 'href="#overview"')
        body = re.sub(r"\[([ACI]\d\d(?:[–, -]+[ACI]\d\d)*)\]", lambda m: f'<a class="source-ref" href="#sources-{m[1][:3]}">[{m[1]}]</a>', body)
        headings, used = [], set()
        def heading(match):
            level, content = match[1], match[2]
            plain = html.unescape(re.sub('<.*?>', '', content))
            source_id = re.match(r'^([ACI]\d\d)\b', plain) if key == 'sources' else None
            ident = f'{key}-{source_id[1]}' if source_id else f'{key}-s{len(used)+1}'
            if ident in used:
                ident += f'-{len(used)+1}'
            used.add(ident)
            if level == '2':
                headings.append((ident, plain))
            return f'<h{level} id="{ident}">{content}</h{level}>'
        body = re.sub(r'<h([23])>(.*?)</h\1>', heading, body)
        toc = '<details class="toc"><summary>Оглавление раздела</summary><ul>' + ''.join(f'<li><a href="#{ident}">{html.escape(label)}</a></li>' for ident, label in headings) + '</ul></details>'
        body = body.replace('</h1>', '</h1>'+toc, 1)
        articles.append(f'<article id="{key}" aria-label="{html.escape(title)}" {"hidden" if key != "overview" else ""}>{body}</article>')
    if len(diagrams) != 19:
        raise ValueError(f"Expected 19 diagrams, got {len(diagrams)}")
    nav = ''.join(f'<a href="#{key}" {"aria-current=page" if key == "overview" else ""}>{title}</a>' for key, title, _ in PAGES)
    script = """
const panels=[...document.querySelectorAll('article')];
function route(){const id=location.hash.slice(1)||'overview',target=document.getElementById(id),panel=target?.closest('article');if(!panel)return;for(const p of panels)p.hidden=p!==panel;for(const a of document.querySelectorAll('nav a')){if(a.hash==='#'+panel.id)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');}if(target===panel)window.scrollTo(0,0);else requestAnimationFrame(()=>target.scrollIntoView({block:'start'}))}
window.addEventListener('hashchange',route);route();
const modal=document.querySelector('dialog'),large=document.getElementById('large-diagram');
for(const b of document.querySelectorAll('button.zoom'))b.addEventListener('click',()=>{const im=b.querySelector('img');large.src=im.src;large.alt=im.alt;large.style.width=Math.max(im.naturalWidth,1100)+'px';modal.showModal()});
document.getElementById('close-zoom').addEventListener('click',()=>modal.close());
modal.addEventListener('click',e=>{if(e.target===modal)modal.close()});
"""
    document = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>AirLLM и Colibrì — атлас оригинальных топологий</title><style>{STYLE}</style></head><body>
<header><div class="eyebrow">AirColibri / исходные механизмы / 13.11.2026</div><h1>AirLLM и Colibrì<br>от запуска до освобождения памяти</h1><p>Два оригинала отдельно, функциональное наложение и план объединения во времени. Данные, управление, владельцы и состояние. 19 схем с увеличением; все изображения встроены в этот файл.</p><p>Статический разбор закреплённых исходников. Runtime-проверки интеграции вынесены отдельно.</p></header>
<nav aria-label="Разделы атласа">{nav}</nav><main>{''.join(articles)}</main>
<dialog aria-label="Увеличенная схема"><button id="close-zoom">Закрыть · Esc</button><div class="zoom-area"><img id="large-diagram" alt=""></div></dialog>
<footer>Основная папка: C:\\air-colibri · без копий проектов · исходники и история сохранены · оформление работает без сети; внешние source links требуют интернет.</footer>
<script>{script}</script></body></html>'''
    output = DOCS / "ORIGINALS_TOPOLOGY_RU.html"
    output.write_text(document, encoding="utf-8")
    receipt = dict(status="RENDERED", renderer="@mermaid-js/mermaid-cli 12.2.0", diagram_count=len(diagrams),
                   diagrams=diagrams, html_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                   note="Rendered source diagrams, not runtime execution evidence.")
    (DOCS / "evidence/2026-11-13/topology-render.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"diagrams": len(diagrams), "html_bytes": output.stat().st_size}))


if __name__ == "__main__":
    build()
