"""Shared SVG primitives for the figures and diagrams in this repo.

No plotting library. Every figure is emitted as SVG text so it stays crisp at any
zoom and legible at 75% browser zoom, where a 1000-wide viewBox scaled into
GitHub's ~890px content column would shrink 10px type below readability.
Canvas width is 900 so the render is close to 1:1 and the minimum type is 15px.
"""

W = 900
BG0, BG1, BG2 = "#12161c", "#0c1013", "#080a0d"
DOT = "#161d26"
NODE_A, NODE_B = "#19212b", "#111820"
STROKE = "#3a4552"
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
BLUE_T, ORANGE_T, AQUA_T = "#6fa8ec", "#e0763f", "#35b183"
INK, INK2, INK3 = "#f3f6f9", "#c8d2dc", "#9aa5b1"
MUTE, FAINT = "#7d8896", "#5c6773"
GRID = "#1e2731"
SANS = "ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"


def head(h, uid, alt):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" role="img" aria-label="{alt}" font-family="{SANS}">
<defs>
  <linearGradient id="bg{uid}" x1="0" y1="0" x2="0.75" y2="1">
    <stop offset="0" stop-color="{BG0}"/><stop offset="0.55" stop-color="{BG1}"/><stop offset="1" stop-color="{BG2}"/>
  </linearGradient>
  <pattern id="dot{uid}" width="28" height="28" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="0.7" fill="{DOT}"/></pattern>
  <linearGradient id="nd{uid}" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="{NODE_A}"/><stop offset="1" stop-color="{NODE_B}"/>
  </linearGradient>
  <linearGradient id="rl{uid}" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="{BLUE}" stop-opacity="0.95"/><stop offset="1" stop-color="{BLUE}" stop-opacity="0"/>
  </linearGradient>
  <marker id="ar{uid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#5a6673"/></marker>
  <marker id="arg{uid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{AQUA}"/></marker>
  <marker id="aro{uid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{ORANGE}"/></marker>
  <marker id="arb{uid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{BLUE}"/></marker>
</defs>
<rect x="0" y="0" width="{W}" height="{h}" fill="url(#bg{uid})"/>
<rect x="0" y="0" width="{W}" height="{h}" fill="url(#dot{uid})"/>
'''


def title_block(uid, eyebrow, title, x=36):
    return (f'<text x="{x}" y="42" fill="{BLUE}" font-size="15" font-weight="700" letter-spacing="3" font-family="{MONO}">{eyebrow}</text>\n'
            f'<text x="{x}" y="78" fill="{INK}" font-size="25" font-weight="700">{title}</text>\n'
            f'<rect x="{x}" y="92" width="130" height="2.5" fill="url(#rl{uid})"/>\n')


def txt(x, y, s, size=15, fill=INK3, anchor="middle", mono=True, weight="400"):
    fam = MONO if mono else SANS
    return f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" font-family="{fam}">{s}</text>\n'


def box(x, y, w, h, uid, stroke=STROKE, sw=1.5, fill=None):
    f = fill or f"url(#nd{uid})"
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{f}" stroke="{stroke}" stroke-width="{sw}" rx="4"/>\n'


def caption(lines, y0, x=36, size=15):
    out = ""
    for i, l in enumerate(lines):
        out += f'<text x="{x}" y="{y0 + i * 22}" fill="{MUTE}" font-size="{size}" font-family="{SANS}">{l}</text>\n'
    return out


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    s = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / d
    return c - s, c + s
