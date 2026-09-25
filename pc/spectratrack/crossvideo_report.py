from __future__ import annotations

from html import escape
import json
from pathlib import Path


def _image_src(preview_path: str | None, report_dir: Path) -> str:
    if not preview_path:
        return ""
    path = Path(preview_path)
    try:
        return escape(path.resolve().relative_to(report_dir.resolve()).as_posix())
    except ValueError:
        return escape(path.resolve().as_uri())


def write_html_report(graph: dict, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tracklets = {t["key"]: t for t in graph.get("tracklets", [])}
    entities_html = []
    for entity in graph.get("entities", []):
        members = []
        for member in entity.get("members", []):
            src = _image_src(member.get("preview_path"), output.parent)
            image = f'<img src="{src}" loading="lazy">' if src else '<div class="noimg">no preview</div>'
            members.append(
                '<div class="member">'
                + image
                + f'<b>{escape(member["video"])}</b>'
                + f'<span>T{int(member["local_track_id"]):03d}</span>'
                + f'<span>{member.get("start_s")}s → {member.get("end_s")}s</span>'
                + f'<span>obs {member.get("observations")}</span>'
                + '</div>'
            )
        entities_html.append(
            '<section class="entity">'
            + f'<h2>{escape(entity["entity_id"])} <small>{escape(entity["label"])}</small></h2>'
            + f'<p>{escape(entity["semantics"])}</p>'
            + '<div class="members">'
            + "".join(members)
            + '</div></section>'
        )

    edge_cards = []
    for edge in graph.get("edges", []):
        left = tracklets.get(edge["left"], {})
        right = tracklets.get(edge["right"], {})
        left_src = _image_src(left.get("preview_path"), output.parent)
        right_src = _image_src(right.get("preview_path"), output.parent)
        left_img = f'<img src="{left_src}" loading="lazy">' if left_src else '<div class="noimg">no preview</div>'
        right_img = f'<img src="{right_src}" loading="lazy">' if right_src else '<div class="noimg">no preview</div>'
        edge_key = f'{edge["left"]}|{edge["right"]}'
        stored = edge.get("review_decision")
        same_label = "LOOKS SAME" if edge.get("relation") == "same_appearance_candidate" else "SAME OBJECT"
        similarity = edge.get("similarity")
        similarity_text = f"{float(similarity):.3f}" if similarity is not None else "N/A"
        edge_cards.append(
            f'<section class="edge {escape(edge["strength"])}" data-key="{escape(edge_key)}">'
            + f'<h3>{escape(edge["strength"].upper())} · similarity {similarity_text}</h3>'
            + f'<p>{escape(edge["relation"])}</p>'
            + '<div class="pair">'
            + f'<div>{left_img}<b>{escape(edge["left"])}</b></div>'
            + f'<div>{right_img}<b>{escape(edge["right"])}</b></div>'
            + '</div>'
            + '<div class="actions">'
            + f'<button onclick="decide(this,\'same\')">{same_label}</button>'
            + '<button onclick="decide(this,\'different\')">DIFFERENT</button>'
            + '<button onclick="decide(this,\'unsure\')">UNSURE</button>'
            + f'<span class="decision">{escape(str(stored).upper()) if stored else ""}</span>'
            + '</div></section>'
        )

    safety = graph.get("safety_semantics", {})
    meta = json.dumps(
        {
            "descriptor": graph.get("descriptor"),
            "thresholds": graph.get("thresholds"),
            "run": graph.get("run", {}),
        },
        ensure_ascii=False,
        indent=2,
    )
    document = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SpectraTrack Cross-Video Review</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#111;color:#eee;margin:24px;line-height:1.35}}
h1,h2,h3{{margin:.3em 0}} small{{color:#aaa;font-weight:400}} p{{color:#bbb}}
.grid,.members,.pair{{display:flex;gap:12px;flex-wrap:wrap}}
.entity,.edge{{border:1px solid #444;border-radius:10px;padding:14px;margin:14px 0;background:#181818}}
.edge.review{{border-color:#777}} .edge.strong{{border-color:#aaa}}
.member,.pair>div{{display:flex;flex-direction:column;gap:4px;background:#202020;padding:8px;border-radius:8px}}
img,.noimg{{width:220px;height:150px;object-fit:contain;background:#090909;border-radius:6px}}
.noimg{{display:grid;place-items:center;color:#666}}
button{{margin:6px 6px 0 0;padding:8px 11px;background:#292929;color:#eee;border:1px solid #666;border-radius:6px}}
button:hover{{background:#3a3a3a}} .decision{{margin-left:8px;font-weight:600}}
pre{{white-space:pre-wrap;background:#181818;padding:12px;border-radius:8px}}
</style>
</head>
<body>
<h1>SpectraTrack Cross-Video Review</h1>
<p><b>People:</b> {escape(safety.get("person", ""))}</p>
<p><b>Other classes:</b> {escape(safety.get("other_classes", ""))}</p>
<button onclick="exportReview()">EXPORT REVIEW JSON</button>
<h2>Entities</h2>
{"".join(entities_html) or "<p>No entities.</p>"}
<h2>Candidate links</h2>
{"".join(edge_cards) or "<p>No candidate links above threshold.</p>"}
<h2>Run metadata</h2>
<pre>{escape(meta)}</pre>
<script>
function decide(button, value) {{
  const card = button.closest('.edge');
  const key = 'spectratrack:' + card.dataset.key;
  localStorage.setItem(key, value);
  card.querySelector('.decision').textContent = value.toUpperCase();
}}
function exportReview() {{
  const decisions = [];
  document.querySelectorAll('.edge').forEach(card => {{
    const pair = card.dataset.key.split('|');
    const stored = localStorage.getItem('spectratrack:' + card.dataset.key);
    const shown = card.querySelector('.decision').textContent.toLowerCase();
    const value = stored || shown;
    if (value === 'same' || value === 'different' || value === 'unsure') {{
      decisions.push({{left: pair[0], right: pair[1], decision: value}});
    }}
  }});
  const payload = JSON.stringify({{version: 1, decisions: decisions}}, null, 2);
  const blob = new Blob([payload], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'spectratrack_review.json';
  link.click();
  URL.revokeObjectURL(url);
}}
document.querySelectorAll('.edge').forEach(card => {{
  const value = localStorage.getItem('spectratrack:' + card.dataset.key);
  if (value) card.querySelector('.decision').textContent = value.toUpperCase();
}});
</script>
</body>
</html>"""
    output.write_text(document, encoding="utf-8")
    return output
