#!/usr/bin/env python3
"""Render the sweep result straight from paper/results/sweep.json."""
import json, os, sys
from statistics import mean, stdev
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgkit import *  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "paper", "results", "sweep.json")))
runs = d["runs"]
Q = [("sparse-sparse", "self sparse\npeer sparse"), ("dense-sparse", "self DENSE\npeer sparse"),
     ("sparse-dense", "self sparse\npeer DENSE"), ("dense-dense", "self DENSE\npeer DENSE")]

H = 560
s = head(H, "sw", "Four-quadrant reward density sweep: the self axis moves the task metric, the peer axis does not, and the predicted drop in stranding does not appear")
s += title_block("sw", "SWEEP RESULT", "The self axis moves it. The peer axis does not.")

x0, x1, ytop, ybot = 120, 470, 150, 380
vals = {k: [r["drones_home"] for r in runs[k]] for k, _ in Q}
top = 2.2
s += txt(x0, 134, "mean drones home, 4 possible", 14, INK3, anchor="start")
for g in (0, 1, 2):
    y = ybot - (ybot - ytop) * g / top
    s += f'<line x1="{x0}" y1="{y:.0f}" x2="{x1}" y2="{y:.0f}" stroke="{GRID}" stroke-width="1"/>\n'
    s += txt(x0 - 12, y + 5, str(g), 14, MUTE, anchor="end")
slot = (x1 - x0) / 4
for i, (k, lab) in enumerate(Q):
    m, sd = mean(vals[k]), stdev(vals[k])
    cx = x0 + slot * (i + 0.5)
    by = ybot - (ybot - ytop) * m / top
    col = AQUA if "dense-" == k[:6] else BLUE
    col = AQUA if k.startswith("dense") else BLUE
    s += f'<rect x="{cx-26:.0f}" y="{by:.0f}" width="52" height="{ybot-by:.0f}" fill="{col}" rx="3"/>\n'
    ylo, yhi = ybot - (ybot-ytop)*(m-sd)/top, ybot - (ybot-ytop)*(m+sd)/top
    s += (f'<line x1="{cx:.0f}" y1="{yhi:.0f}" x2="{cx:.0f}" y2="{ylo:.0f}" stroke="{INK2}" stroke-width="2"/>\n'
          f'<line x1="{cx-9:.0f}" y1="{yhi:.0f}" x2="{cx+9:.0f}" y2="{yhi:.0f}" stroke="{INK2}" stroke-width="2"/>\n'
          f'<line x1="{cx-9:.0f}" y1="{ylo:.0f}" x2="{cx+9:.0f}" y2="{ylo:.0f}" stroke="{INK2}" stroke-width="2"/>\n')
    s += txt(cx, yhi - 12, f"{m:.2f}", 17, INK, weight="700")
    for j, line in enumerate(lab.split("\n")):
        s += txt(cx, ybot + 26 + j * 17, line, 12, INK3)
s += f'<line x1="{x0}" y1="{ybot}" x2="{x1}" y2="{ybot}" stroke="#46525f" stroke-width="1.5"/>\n'

bx = 540
s += txt(bx, 134, "what the sweep says", 14, INK3, anchor="start")
rows = [("self axis", "+0.48 and +0.46", "p = 0.013, 0.002", AQUA, "moves the metric"),
        ("peer axis", "-0.07 and -0.09", "p = 0.077, 0.511", ORANGE, "no effect"),
        ("stranding", "+0.07 in dense/dense", "p = 0.039", ORANGE, "predicted to fall; rose")]
for i, (name, eff, p, col, note) in enumerate(rows):
    y = 168 + i * 74
    s += f'<rect x="{bx}" y="{y}" width="324" height="60" fill="url(#ndsw)" stroke="{col}" stroke-width="1.5" rx="4"/>\n'
    s += txt(bx + 14, y + 24, name, 15, col, anchor="start", weight="700")
    s += txt(bx + 310, y + 24, eff, 15, INK, anchor="end")
    s += txt(bx + 14, y + 46, note, 13, MUTE, anchor="start")
    s += txt(bx + 310, y + 46, p, 13, FAINT, anchor="end")

s += f'<rect x="{bx}" y="390" width="324" height="66" fill="#1c130e" stroke="{ORANGE}" stroke-width="1.5" rx="4"/>\n'
s += txt(bx + 162, 414, "why the peer axis is inert here", 14, ORANGE_T, weight="700")
s += txt(bx + 162, 436, "clearance spanned 0.946 to 0.970 over", 13, INK3)
s += txt(bx + 162, 452, "all 20 runs, so it carries no gradient", 13, INK3)

s += caption(["Five seeds per cell, 2000 training episodes, 100 greedy evaluation episodes, Welch t-tests. Only the reward-density",
              "switches differ across cells. The self-axis effect replicates potential-based shaping and is not new. The peer-axis",
              "prediction is not supported, and the environment is the reason: it cannot generate the contention the axis is about."], 486)
open(os.path.join(ROOT, "paper", "results", "sweep.svg"), "w").write(s + "</svg>\n")
print("wrote paper/results/sweep.svg")
