from __future__ import annotations

import json
from string import Template

from property_guardian_ai.models.schemas import ComponentRisk, PropertyRecord, RiskAssessment

# Status palette (never reused for series); always paired with a text label.
BAND_STYLE: dict[str, tuple[str, str, str]] = {  # band -> (fill, text-on-fill, icon)
    "Low": ("#0ca30c", "#ffffff", "✓"),
    "Moderate": ("#fab219", "#0b0b0b", "!"),
    "High": ("#ec835a", "#0b0b0b", "▲"),
    "Very High": ("#d03b3b", "#ffffff", "▲▲"),
}
# Where each component's score badge sits on the drawing (SVG user units).
BADGE_POSITIONS: dict[str, tuple[int, int]] = {
    "roof": (320, 140), "gutters": (500, 214), "exterior": (205, 305), "openings": (408, 248),
    "foundation": (232, 354), "plumbing": (372, 370), "yard": (70, 172),
}
NEAR_WATER_KM = 2.0
RAINY_EVENTS = 10


def _badge(component: ComponentRisk) -> str:
    x, y = BADGE_POSITIONS[component.key]
    fill, ink, _ = BAND_STYLE[component.band]
    return (
        f'<g class="badge" data-key="{component.key}" transform="translate({x},{y})">'
        f'<circle r="17" fill="{fill}" stroke="#fff" stroke-width="2"/>'
        f'<text text-anchor="middle" dy="4.5" fill="{ink}">{component.score:.0f}</text></g>'
    )


def _payload(assessment: RiskAssessment) -> dict[str, object]:
    return {
        "components": [
            {**c.model_dump(mode="json"), "style": BAND_STYLE[c.band]} for c in assessment.component_risks
        ],
        "default": max(assessment.component_risks, key=lambda c: c.score).key,
    }


def _scene_extras(record: PropertyRecord) -> str:
    extras = []
    if record.weather.heavy_rainfall_events >= RAINY_EVENTS:
        rain = "".join(f'<line x1="{x}" y1="74" x2="{x - 8}" y2="96"/>' for x in range(452, 540, 14))
        extras.append(
            '<g class="cloud"><ellipse cx="490" cy="52" rx="52" ry="22"/><ellipse cx="458" cy="60" rx="30" ry="18"/>'
            f'<ellipse cx="528" cy="62" rx="30" ry="16"/><g class="rain">{rain}</g></g>'
        )
    if record.flood.distance_to_water_km < NEAR_WATER_KM:
        extras.append(
            '<path class="water" d="M520 330 q15 -8 30 0 t30 0 t30 0 t30 0 V460 H520 Z"/>'
            f'<text class="scene-label" x="580" y="356" text-anchor="middle">Water · '
            f'{record.flood.distance_to_water_km:g} km</text>'
        )
    return "".join(extras)


HOUSE_TEMPLATE = Template(r"""
<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  :root { --ink:#0b0b0b; --ink-2:#52514e; --line:#e2e0da; --surface:#fcfcfb; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.45 system-ui,-apple-system,sans-serif; color:var(--ink); background:transparent; }
  .wrap { display:flex; gap:20px; flex-wrap:wrap; }
  .card { background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:16px; }
  .house { flex:1 1 520px; min-width:340px; }
  .panel { flex:1 1 340px; min-width:300px; max-height:600px; overflow:auto; }
  .addr { font-weight:650; font-size:15px; }
  .facts { color:var(--ink-2); font-size:12.5px; margin:2px 0 8px; }
  .toolbar { display:flex; justify-content:space-between; align-items:center; font-size:12.5px; color:var(--ink-2); }
  svg { width:100%; height:auto; display:block; }
  .part { cursor:pointer; outline:none; }
  .part .shape { transition: fill .15s ease, stroke-width .1s ease; }
  .part:hover .shape, .part:focus-visible .shape { stroke:#0b0b0b; stroke-width:3; }
  .part.selected .shape { stroke:#2a78d6; stroke-width:4; }
  .heat .part .shape { fill:var(--heat) !important; fill-opacity:.8; }
  .heat .part .pipe { stroke:var(--heat) !important; }
  .hit { stroke:transparent; stroke-width:22; fill:none; }
  .badge { cursor:pointer; font:700 12.5px system-ui,sans-serif; }
  .badge:hover circle { stroke:#0b0b0b; }
  .cloud ellipse { fill:#c9ced6; } .rain line { stroke:#5598e7; stroke-width:2; stroke-linecap:round; }
  .water { fill:#9ec5f4; } .scene-label { font:600 11px system-ui,sans-serif; fill:#184f95; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
  .chip { border:1px solid var(--line); background:#fff; border-radius:999px; padding:4px 10px; font-size:12.5px;
          cursor:pointer; display:flex; gap:6px; align-items:center; }
  .chip[aria-pressed="true"] { border-color:#2a78d6; box-shadow:0 0 0 2px #2a78d633; }
  .dot { width:9px; height:9px; border-radius:50%; display:inline-block; }
  h2 { margin:0; font-size:20px; } h3 { margin:16px 0 6px; font-size:13px; text-transform:uppercase;
       letter-spacing:.04em; color:var(--ink-2); }
  .scoreline { display:flex; align-items:baseline; gap:10px; margin:6px 0 4px; }
  .big { font-size:34px; font-weight:700; letter-spacing:-.02em; }
  .pill { border-radius:999px; padding:2px 10px; font-weight:650; font-size:12.5px; }
  .bar { height:8px; border-radius:4px; background:#f0efec; overflow:hidden; margin-bottom:8px; }
  .bar > div { height:100%; border-radius:4px; }
  ul { margin:0; padding-left:18px; } li { margin:3px 0; }
  .rec { border:1px solid var(--line); border-radius:12px; padding:10px 12px; margin:8px 0; background:#fff; }
  .rec .t { font-weight:650; } .muted { color:var(--ink-2); font-size:12.5px; }
  .contrib { display:flex; justify-content:space-between; gap:8px; font-size:13px; padding:3px 0;
             border-bottom:1px dashed var(--line); }
  .up { color:#b42323; font-weight:650; } .down { color:#1c5cab; font-weight:650; }
</style></head><body>
<div class="wrap">
  <div class="card house">
    <div class="addr">$address</div>
    <div class="facts">$facts</div>
    <div class="toolbar"><span>Click any part of the house to inspect its risk.</span>
      <label><input type="checkbox" id="heat"> Risk heatmap</label></div>
    <svg id="scene" viewBox="0 0 640 460" role="img" aria-label="Interactive house diagram">
      <rect x="0" y="0" width="640" height="330" fill="#eef4fb"/>
      <rect x="0" y="330" width="640" height="130" fill="#e3ecd9"/>
      $extras
      <g class="part" data-key="yard" tabindex="0" role="button" aria-label="Yard and trees">
        <title>Yard & trees</title>
        <rect class="shape" x="62" y="250" width="16" height="82" fill="#7b5a3c"/>
        <circle class="shape" cx="70" cy="228" r="44" fill="#5c9a4b"/>
        <circle class="shape" cx="42" cy="258" r="28" fill="#6aa857"/>
        <circle class="shape" cx="100" cy="258" r="28" fill="#6aa857"/>
        <ellipse class="shape" cx="138" cy="326" rx="26" ry="12" fill="#6aa857"/>
      </g>
      <g class="part" data-key="foundation" tabindex="0" role="button" aria-label="Foundation and basement">
        <title>Foundation & basement</title>
        <rect class="shape" x="170" y="330" width="300" height="64" fill="#d6d0c4" stroke="#8a8170" stroke-width="1.5"/>
      </g>
      <g class="part" data-key="plumbing" tabindex="0" role="button" aria-label="Plumbing">
        <title>Plumbing</title>
        <path class="hit" d="M120 378 H300 V348 H425"/>
        <path class="shape pipe" d="M120 378 H300 V348 H425" fill="none" stroke="#3987e5" stroke-width="6"
              stroke-linecap="round" stroke-linejoin="round"/>
        <rect class="shape" x="425" y="336" width="28" height="50" rx="5" fill="#b7d3f6" stroke="#256abf" stroke-width="1.5"/>
      </g>
      <g class="part" data-key="exterior" tabindex="0" role="button" aria-label="Exterior walls and siding">
        <title>Exterior walls & siding</title>
        <rect class="shape" x="180" y="198" width="280" height="132" fill="#f3e9d8" stroke="#8a7a63" stroke-width="1.5"/>
      </g>
      <g class="part" data-key="roof" tabindex="0" role="button" aria-label="Roof">
        <title>Roof</title>
        <rect class="shape" x="398" y="106" width="22" height="48" fill="#6b4f42"/>
        <polygon class="shape" points="160,198 320,92 480,198" fill="#7a5c4c" stroke="#4d382e" stroke-width="1.5"/>
      </g>
      <g class="part" data-key="gutters" tabindex="0" role="button" aria-label="Gutters and drainage">
        <title>Gutters & drainage</title>
        <rect class="shape" x="154" y="196" width="332" height="9" rx="3" fill="#9aa3ad"/>
        <rect class="shape" x="468" y="205" width="8" height="125" fill="#9aa3ad"/>
        <rect class="shape" x="468" y="324" width="34" height="7" rx="3" fill="#9aa3ad"/>
      </g>
      <g class="part" data-key="openings" tabindex="0" role="button" aria-label="Doors and windows">
        <title>Doors & windows</title>
        <rect class="shape" x="205" y="226" width="56" height="46" rx="3" fill="#cfe3f5" stroke="#6b7c8f" stroke-width="2"/>
        <rect class="shape" x="380" y="226" width="56" height="46" rx="3" fill="#cfe3f5" stroke="#6b7c8f" stroke-width="2"/>
        <rect class="shape" x="295" y="258" width="50" height="72" rx="3" fill="#5b7fa6" stroke="#3d566f" stroke-width="2"/>
      </g>
      $badges
    </svg>
    <div class="chips" id="chips"></div>
  </div>
  <div class="card panel" id="panel" aria-live="polite"></div>
</div>
<script>
const DATA = $payload;
const byKey = Object.fromEntries(DATA.components.map(c => [c.key, c]));
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k === "style" ? n.style.cssText = v : n.setAttribute(k, v);
  for (const kid of kids) n.append(kid);
  return n;
};
document.querySelectorAll(".part").forEach(p => {
  const c = byKey[p.dataset.key]; if (c) p.style.setProperty("--heat", c.style[0]);
});

function render(key) {
  const c = byKey[key]; if (!c) return;
  const [fill, ink, icon] = c.style;
  document.querySelectorAll(".part").forEach(p => p.classList.toggle("selected", p.dataset.key === key));
  document.querySelectorAll(".chip").forEach(ch => ch.setAttribute("aria-pressed", ch.dataset.key === key));
  const panel = document.getElementById("panel"); panel.replaceChildren();
  panel.append(
    el("h2", {}, c.name),
    el("div", {class: "scoreline"}, el("span", {class: "big"}, c.score.toFixed(0)), el("span", {class: "muted"}, "/ 100"),
       el("span", {class: "pill", style: `background:$${fill};color:$${ink}`}, `$${icon} $${c.band}`)),
    el("div", {class: "bar"}, el("div", {style: `width:$${c.score}%;background:$${fill}`})),
    el("div", {class: "muted"}, c.explanation),
    el("h3", {}, "Why — evidence"),
    el("ul", {}, ...c.evidence.map(e => el("li", {}, e))),
  );
  if (c.model_contributions.length) {
    panel.append(el("h3", {}, "Model attribution (SHAP)"));
    c.model_contributions.forEach(m => {
      const cls = m.contribution > 0 ? "up" : "down";
      const sim = m.provenance === "simulated" ? " (simulated)" : "";
      panel.append(el("div", {class: "contrib"}, el("span", {}, `$${m.label} = $${m.value}$${sim}`),
        el("span", {class: cls}, `$${m.contribution > 0 ? "+" : ""}$${m.contribution.toFixed(1)} pts`)));
    });
  }
  panel.append(el("h3", {}, "What reduces this risk"));
  if (!c.recommendations.length) panel.append(el("div", {class: "muted"}, "No action triggered for this component."));
  c.recommendations.forEach(r => panel.append(el("div", {class: "rec"},
    el("div", {class: "t"}, r.title), el("div", {}, r.description),
    el("div", {class: "muted"}, `Priority: $${r.priority} · Evidence: $${r.evidence}`))));
  panel.append(el("p", {class: "muted"}, "Component scores are transparent heuristic indices; SHAP values come from the validated model and are shown for comparison."));
}

const chips = document.getElementById("chips");
[...DATA.components].sort((a, b) => b.score - a.score).forEach(c => {
  const chip = el("button", {class: "chip", "data-key": c.key, "aria-pressed": "false"},
    el("span", {class: "dot", style: `background:$${c.style[0]}`}), `$${c.name} · $${c.score.toFixed(0)} $${c.band}`);
  chip.onclick = () => render(c.key); chips.append(chip);
});
document.querySelectorAll(".part, .badge").forEach(p => {
  p.addEventListener("click", () => render(p.dataset.key));
  p.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); render(p.dataset.key); } });
});
document.getElementById("heat").onchange = e => document.getElementById("scene").classList.toggle("heat", e.target.checked);
render(DATA.default);
</script></body></html>
""")


def build_house_html(assessment: RiskAssessment, record: PropertyRecord) -> str:
    profile = assessment.property_profile
    facts = (
        f"Built {profile.year_built} · roof {profile.roof_age} yrs · {profile.square_footage:,} sq ft · "
        f"flood zone {record.flood.flood_zone} · overall risk {assessment.risk_score:.0f} ({assessment.risk_label})"
    )
    payload = json.dumps(_payload(assessment)).replace("</", "<\\/")
    return HOUSE_TEMPLATE.substitute(
        address=_escape(profile.address.split("(")[0].strip()),
        facts=_escape(facts),
        extras=_scene_extras(record),
        badges="".join(_badge(c) for c in assessment.component_risks),
        payload=payload,
    )


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
