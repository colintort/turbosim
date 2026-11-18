"""PDF reports for POST /report.

The page (frontend/src/report/collect.js) gathers what is on screen: the
setup, a sketch of the cycle, the result tiles, the tables, the T-s / P-h /
P-v data and the solver's notes. Everything arrives already in display units
and formatted the way the app shows it. This module only lays it out on
Letter pages, so the report always matches the screen.

Payload:
    { title, version, mode, facts: [[label, value]], sections: [section, ...] }

Section kinds:
    stats   { title, items: [{ sym, sub, value, unit, note, accent }] }
    table   { title, columns: [str], rows: [[str]], solved: [[bool]] (optional), note }
    facts   { title, items: [[label, value]] }
    sketch  { title, devices: [{ x, y, w, h, label, letter }],
              pipes: [{ path (SVG d, flow coordinates), label, lx, ly, color }] }
    plots   { title, plots: [{ title, x, y, logX, logY, dome: [[x, y]],
              states: { n: [x, y] }, paths: [{ kind, points: [[x, y]] }] }] }
    notes   { title, lines: [str] }
    work    { title, source, notes: [str], items: [{ title, kind, steps: [{ head, lines: [[cls, text]] }] }] }
            the worked solution from workings.py (View > Show my work)

A section that can't be drawn is replaced by a one-line note, so one bad
part never costs the whole report.
"""
import io
import math
import os
import re
import sys
import time
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Circle, Drawing, Group, Line, PolyLine, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.platypus import (CondPageBreak, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)


class ReportError(ValueError):
    pass


PAGE = letter
MARGIN = 0.6 * inch
FRAME_W = PAGE[0] - 2 * MARGIN

INK = colors.HexColor("#1b2233")
MUTED = colors.HexColor("#6b7489")
RULE = colors.HexColor("#d5dbe8")
TILE = colors.HexColor("#f3f5fa")
TILE_ACCENT = colors.HexColor("#e6f0ff")
SOLVED = colors.HexColor("#1d5fd1")       # solved values, blue as on screen
HEAD = colors.HexColor("#22305a")

# What crosses the boundary along a process line (Colin's colors; yellow is
# darkened so it reads on white paper).
KIND_COLOR = {
    "Qin": colors.HexColor("#d62828"),
    "Qout": colors.HexColor("#1d4ed8"),
    "Win": colors.HexColor("#7c3aed"),
    "Wout": colors.HexColor("#c99a06"),
}
KIND_TEXT = {"Qin": "heat in", "Qout": "heat out", "Win": "work in", "Wout": "work out"}
PATH_COLOR = colors.HexColor("#3a4256")
DOME_COLOR = colors.HexColor("#9aa3b2")


# ---------- fonts ----------

_fonts = None


def _font_dirs():
    here = os.path.dirname(os.path.abspath(__file__))
    bundled = getattr(sys, "_MEIPASS", None)     # inside TurboSim.exe
    return [d for d in (bundled and os.path.join(bundled, "fonts"), os.path.join(here, "fonts")) if d]


def fonts():
    """(regular, bold) font names. The bundled DejaVu Sans has η, ṁ, Ẇ, Q̇
    and the rest; Windows fonts come next; Helvetica is the last resort."""
    global _fonts
    if _fonts:
        return _fonts
    win = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    pairs = [(os.path.join(d, "DejaVuSans.ttf"), os.path.join(d, "DejaVuSans-Bold.ttf")) for d in _font_dirs()]
    pairs += [(os.path.join(win, "segoeui.ttf"), os.path.join(win, "segoeuib.ttf")),
              (os.path.join(win, "arial.ttf"), os.path.join(win, "arialbd.ttf"))]
    for regular, bold in pairs:
        if not (os.path.exists(regular) and os.path.exists(bold)):
            continue
        try:
            pdfmetrics.registerFont(TTFont("Report", regular))
            pdfmetrics.registerFont(TTFont("Report-Bold", bold))
        except Exception:
            continue
        for italic in (0, 1):
            addMapping("Report", 0, italic, "Report")
            addMapping("Report", 1, italic, "Report-Bold")
        _fonts = ("Report", "Report-Bold")
        return _fonts
    _fonts = ("Helvetica", "Helvetica-Bold")
    return _fonts


# Helvetica only has Latin-1; spell out what it can't draw.
_PLAIN = {"η": "eta", "ṁ": "m", "Ẇ": "W", "Q̇": "Q", "V̇": "V", "ω": "omega", "λ": "lambda", "ε": "eps",
          "Δ": "d", "Σ": "sum", "π": "pi", "−": "-", "–": "-", "—": "-", "·": "·", "₁": "1", "₂": "2",
          "≈": "~", "→": "->", "≤": "<=", "≥": ">="}


def _txt(s):
    s = "" if s is None else str(s)
    if fonts()[0] != "Helvetica":
        return s
    for k, v in _PLAIN.items():
        s = s.replace(k, v)
    s = s.replace("\u0307", "")
    return s.encode("latin-1", "replace").decode("latin-1")


def _esc(s):
    return escape(_txt(s))


# ---------- styles ----------

def _styles():
    regular, bold = fonts()
    base = ParagraphStyle("base", fontName=regular, fontSize=8.5, leading=11, textColor=INK)
    return {
        "base": base,
        "title": ParagraphStyle("title", base, fontName=bold, fontSize=18, leading=22, textColor=HEAD),
        "h2": ParagraphStyle("h2", base, fontName=bold, fontSize=11.5, leading=15, textColor=HEAD,
                             spaceBefore=12, spaceAfter=5),
        "small": ParagraphStyle("small", base, fontSize=7, leading=9, textColor=MUTED),
        "cell": ParagraphStyle("cell", base, fontSize=7.8, leading=9.6),
        "cellHead": ParagraphStyle("cellHead", base, fontName=bold, fontSize=7.8, leading=9.6, textColor=HEAD),
        "sym": ParagraphStyle("sym", base, fontSize=8, leading=10, textColor=MUTED),
        "val": ParagraphStyle("val", base, fontName=bold, fontSize=11.5, leading=14),
        "workTitle": ParagraphStyle("workTitle", base, fontName=bold, fontSize=10, leading=13, textColor=HEAD,
                                    spaceAfter=2),
        "workHead": ParagraphStyle("workHead", base, fontName=bold, fontSize=8.5, leading=11, spaceBefore=3,
                                   spaceAfter=1),
        "workAsm": ParagraphStyle("workAsm", base, textColor=MUTED),
        "workRes": ParagraphStyle("workRes", base, fontName=bold, textColor=SOLVED),
    }


# ---------- sections ----------

def _heading(title, st):
    return [CondPageBreak(1.2 * inch), Paragraph(_esc(title), st["h2"])]


def _facts(items, st):
    rows = [[Paragraph(_esc(k), st["sym"]), Paragraph(_esc(v), st["base"])] for k, v in items if v not in (None, "")]
    if not rows:
        return []
    t = Table(rows, colWidths=[1.25 * inch, FRAME_W - 1.25 * inch], hAlign="LEFT")
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return [t]


def _stats(items, st, per_row=4):
    cells = []
    for it in items:
        sym = _esc(it.get("sym", ""))
        if it.get("sub"):
            sym += f"<sub>{_esc(it['sub'])}</sub>"
        val = _esc(it.get("value", ""))
        if it.get("unit"):
            val += f' <font size="8" color="#6b7489">{_esc(it["unit"])}</font>'
        cell = [Paragraph(sym, st["sym"]), Paragraph(val, st["val"])]
        if it.get("note"):
            cell.append(Paragraph(_esc(it["note"]), st["small"]))
        cells.append((cell, bool(it.get("accent"))))
    if not cells:
        return []
    rows, style = [], [("VALIGN", (0, 0), (-1, -1), "TOP"),
                       ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                       ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]
    for i in range(0, len(cells), per_row):
        chunk = cells[i:i + per_row]
        r = len(rows)
        rows.append([c for c, _ in chunk] + [""] * (per_row - len(chunk)))
        for j, (_, accent) in enumerate(chunk):
            style.append(("BACKGROUND", (j, r), (j, r), TILE_ACCENT if accent else TILE))
            style.append(("BOX", (j, r), (j, r), 2, colors.white))   # gaps between tiles
    t = Table(rows, colWidths=[FRAME_W / per_row] * per_row, hAlign="LEFT")
    t.setStyle(TableStyle(style))
    return [t]


def _col_widths(columns, rows, width):
    """Wide enough for each column's longest entry, scaled to fill the page."""
    est = []
    for j, c in enumerate(columns):
        longest = max([len(str(c))] + [len(str(r[j])) for r in rows if j < len(r)])
        est.append(min(max(longest, 3), 48))
    total = sum(est)
    return [width * e / total for e in est]


def _table(sec, st):
    if not isinstance(sec.get("rows") or [], list) or not all(isinstance(r, list) for r in sec.get("rows") or []):
        raise ReportError("its rows are not a list of rows")
    columns = [str(c) for c in sec.get("columns") or []]
    rows = [[("" if v is None else str(v)) for v in r] for r in sec.get("rows") or []]
    if not columns or not rows:
        return []
    n = len(columns)
    rows = [(r + [""] * n)[:n] for r in rows]
    solved = sec.get("solved") or []
    data = [[Paragraph(_esc(c), st["cellHead"]) for c in columns]]
    for i, r in enumerate(rows):
        mask = solved[i] if i < len(solved) else []
        data.append([Paragraph(f'<font color="#1d5fd1">{_esc(v)}</font>' if j < len(mask) and mask[j] else _esc(v),
                               st["cell"]) for j, v in enumerate(r)])
    t = Table(data, colWidths=_col_widths(columns, rows, FRAME_W), repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.9, HEAD),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fc")]),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    # the note goes above the table, so a long table can't strand it on the next page
    note = sec.get("note")
    if note is None and any(any(m) for m in solved):
        note = "Black: given. Blue: solved."
    return ([Paragraph(_esc(note), st["small"]), Spacer(1, 3)] if note else []) + [t]


def _notes(lines, st):
    return [Paragraph("• " + _esc(line), st["base"]) for line in lines if str(line).strip()]


# ---------- the worked solution (Show my work, 1.10.7) ----------

_WORK_TAG = {"law": "Law", "asm": "Assume", "eq": "Reduce", "num": "Numbers", "res": "Answer", "note": "Note"}


def _work(sec, st):
    """Each device, state and the cycle: its steps, one line per row with the
    kind of line (law, assumption, ...) in a narrow column on the left."""
    out = []
    if sec.get("source"):
        out.append(Paragraph(_esc(f"Written out the way the textbook does it. Equation and table numbers: "
                                  f"{sec['source']}"), st["small"]))
    for n in sec.get("notes") or []:
        out.append(Paragraph(_esc(n), st["small"]))
    tag_w = 0.75 * inch
    for item in sec.get("items") or []:
        block = [Spacer(1, 6), Paragraph(_esc(item.get("title") or ""), st["workTitle"])]
        for s in item.get("steps") or []:
            rows = []
            for cls, text in s.get("lines") or []:
                style = st["workRes"] if cls == "res" else st["workAsm"] if cls in ("asm", "note") else st["base"]
                rows.append([Paragraph(_esc(_WORK_TAG.get(cls, "")), st["small"]), Paragraph(_esc(text), style)])
            if not rows:
                continue
            t = Table(rows, colWidths=[tag_w, FRAME_W - tag_w], hAlign="LEFT")
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                   ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
            block.append(KeepTogether([Paragraph(_esc(s.get("head") or ""), st["workHead"]), t]))
        out += block
    return out


# ---------- the cycle sketch ----------

_NUM = r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def parse_path(d):
    """Points along an SVG path made of M, L, H, V, Q and C commands (what
    React Flow's edge paths use). Curves are sampled; anything else is skipped."""
    pts = []
    cmd, nums = None, []
    need = {"M": 2, "L": 2, "H": 1, "V": 1, "Q": 4, "C": 6}
    for tok in re.findall(rf"[A-Za-z]|{_NUM}", d or ""):
        if tok.isalpha():
            cmd, nums = tok.upper(), []
            if cmd == "Z" and pts:
                pts.append(pts[0])
            continue
        if cmd not in need:
            continue
        nums.append(float(tok))
        if len(nums) < need[cmd]:
            continue
        x0, y0 = pts[-1] if pts else (0.0, 0.0)
        if cmd in ("M", "L"):
            pts.append((nums[0], nums[1]))
        elif cmd == "H":
            pts.append((nums[0], y0))
        elif cmd == "V":
            pts.append((x0, nums[0]))
        elif cmd == "Q":
            cx, cy, x, y = nums
            for k in range(1, 5):
                t = k / 4
                pts.append(((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t * t * x,
                            (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t * t * y))
        elif cmd == "C":
            c1x, c1y, c2x, c2y, x, y = nums
            for k in range(1, 7):
                t = k / 6
                a, b, c, e = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3
                pts.append((a * x0 + b * c1x + c * c2x + e * x, a * y0 + b * c1y + c * c2y + e * y))
        nums = []
        if cmd == "M":
            cmd = "L"            # extra pairs after M are line-tos
    return pts


def _color(css, fallback):
    """A pipe color from the page: "#rrggbb" or "rgb(r, g, b)"."""
    try:
        m = re.match(r"\s*rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", css or "")
        if m:
            return colors.Color(*(min(float(v), 255) / 255 for v in m.groups()))
        return colors.HexColor(css) if css else fallback
    except Exception:
        return fallback


def _device_outline(dr, shapes, css, X, Y):
    """1.10.2: a device drawn as it is on screen (its outlines, sent by the
    page in flow coordinates). The screen's neon colour is darkened so it
    reads on white paper, and closed shapes get a pale tint of it."""
    base = _color(css, colors.HexColor("#5b6478"))
    r, g, b = base.red, base.green, base.blue
    stroke = colors.Color(r * 0.55, g * 0.55, b * 0.55)
    tint = colors.Color(1 - (1 - r) * 0.18, 1 - (1 - g) * 0.18, 1 - (1 - b) * 0.18)
    for s in shapes:                      # fills first, so no outline is covered
        if s.get("closed"):
            flat = [v for q in s["pts"] for v in (X(q[0]), Y(q[1]))]
            dr.add(Polygon(flat, fillColor=tint, strokeColor=None, strokeWidth=0))
    for s in shapes:
        flat = [v for q in s["pts"] for v in (X(q[0]), Y(q[1]))]
        if s.get("closed"):
            flat += flat[:2]
        dr.add(PolyLine(flat, strokeColor=stroke, strokeWidth=1.1, strokeLineJoin=1, strokeLineCap=1,
                        strokeDashArray=[2, 2] if s.get("dash") else None))


def _sketch(sec, max_h=4.2 * inch):
    regular, bold = fonts()
    devices = sec.get("devices") or []
    pipes = [dict(p, pts=parse_path(p.get("path"))) for p in sec.get("pipes") or []]
    xs, ys = [], []
    for d in devices:
        xs += [d["x"], d["x"] + d.get("w", 100)]
        ys += [d["y"], d["y"] + d.get("h", 100) + 18]      # room for the name underneath
        for sh in d.get("shapes") or []:
            xs += [q[0] for q in sh.get("pts") or []]
            ys += [q[1] for q in sh.get("pts") or []]
    for p in pipes:
        xs += [q[0] for q in p["pts"]]
        ys += [q[1] for q in p["pts"]]
    if not xs:
        return None
    pad = 14
    x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    sc = min(FRAME_W / (x1 - x0), max_h / (y1 - y0), 0.9)
    W, H = (x1 - x0) * sc, (y1 - y0) * sc
    dr = Drawing(W, H)
    X = lambda x: (x - x0) * sc             # noqa: E731
    Y = lambda y: H - (y - y0) * sc         # noqa: E731

    for p in pipes:
        pts = p["pts"]
        if len(pts) < 2:
            continue
        col = _color(p.get("color"), colors.HexColor("#2b6cb0"))
        flat = [v for q in pts for v in (X(q[0]), Y(q[1]))]
        dr.add(PolyLine(flat, strokeColor=col, strokeWidth=1.6, strokeLineJoin=1))
        # arrowhead where the pipe enters the device
        (ax, ay), (bx, by) = pts[-2], pts[-1]
        ang = math.atan2(Y(by) - Y(ay), X(bx) - X(ax))
        tx, ty, s = X(bx), Y(by), 5.5
        dr.add(Polygon([tx, ty,
                        tx - s * math.cos(ang - 0.45), ty - s * math.sin(ang - 0.45),
                        tx - s * math.cos(ang + 0.45), ty - s * math.sin(ang + 0.45)],
                       fillColor=col, strokeColor=col, strokeWidth=0.5))

    for d in devices:
        w, h = d.get("w", 100) * sc, d.get("h", 100) * sc
        x, y = X(d["x"]), Y(d["y"]) - h
        shapes = [s for s in d.get("shapes") or [] if len(s.get("pts") or []) > 1]
        if shapes:
            _device_outline(dr, shapes, d.get("color"), X, Y)
        else:
            dr.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=colors.HexColor("#f4f6fb"),
                        strokeColor=colors.HexColor("#5b6478"), strokeWidth=1))
        if d.get("letter"):
            dr.add(String(x + w / 2, y + h / 2 - 5, _txt(d["letter"]), fontName=bold, fontSize=13,
                          fillColor=HEAD, textAnchor="middle"))
        label = _txt(d.get("label", ""))
        if label:
            lw = pdfmetrics.stringWidth(label, regular, 7) + 4
            dr.add(Rect(x + w / 2 - lw / 2, y - 11, lw, 9, fillColor=colors.white, strokeColor=None))
            dr.add(String(x + w / 2, y - 9, label, fontName=regular, fontSize=7, fillColor=INK, textAnchor="middle"))

    for p in pipes:
        if not p.get("label") or p.get("lx") is None:
            continue
        lx, ly = X(p["lx"]), Y(p["ly"])
        text = _txt(p["label"])
        w = pdfmetrics.stringWidth(text, bold, 7) + 7
        col = _color(p.get("color"), colors.HexColor("#2b6cb0"))
        dr.add(Rect(lx - w / 2, ly - 5, w, 10, rx=4, ry=4, fillColor=colors.white, strokeColor=col, strokeWidth=0.9))
        dr.add(String(lx, ly - 2.5, text, fontName=bold, fontSize=7, fillColor=INK, textAnchor="middle"))
    return dr


# ---------- property diagrams ----------

def _nice_ticks(lo, hi, count=5):
    span = hi - lo or 1.0
    raw = span / count
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    out, v = [], math.ceil(lo / step) * step
    while v <= hi + 1e-9 * span:
        out.append(float(f"{v:.10g}"))
        v += step
    return out


def _log_ticks(lo, hi):
    """Tick values (not logs) between 10**lo and 10**hi."""
    decades = hi - lo
    mults = [1] if decades > 4 else [1, 2, 5] if decades > 1.2 else [1, 1.5, 2, 3, 5, 7]
    out = []
    for e in range(math.floor(lo), math.ceil(hi) + 1):
        for m in mults:
            v = m * 10 ** e
            if lo - 1e-9 <= math.log10(v) <= hi + 1e-9:
                out.append(float(f"{v:.6g}"))
    return out


def _fmt_tick(v):
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e5 or a < 1e-3:
        return f"{v:.0e}".replace("e+0", "e").replace("e-0", "e-")
    return f"{v:.6g}"


def _plot(p, W=3.45 * inch, H=2.55 * inch):
    regular, bold = fonts()
    logx, logy = bool(p.get("logX")), bool(p.get("logY"))

    def ok(pt):
        return (pt is not None and len(pt) >= 2 and pt[0] is not None and pt[1] is not None
                and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pt[:2])
                and (not logx or pt[0] > 0) and (not logy or pt[1] > 0))

    def tr(pt):
        return (math.log10(pt[0]) if logx else pt[0], math.log10(pt[1]) if logy else pt[1])

    dome = [tr(q) for q in p.get("dome") or [] if ok(q)]
    paths = [(q.get("kind", ""), [tr(z) for z in q.get("points") or [] if ok(z)]) for q in p.get("paths") or []]
    states = {str(k): tr(v) for k, v in (p.get("states") or {}).items() if ok(v)}
    allpts = dome + [z for _, pts in paths for z in pts] + list(states.values())
    if not allpts:
        return None

    lx, ly = [q[0] for q in allpts], [q[1] for q in allpts]
    xmin, xmax, ymin, ymax = min(lx), max(lx), min(ly), max(ly)
    if xmax - xmin < 1e-12:
        xmin, xmax = xmin - 1, xmax + 1
    if ymax - ymin < 1e-12:
        ymin, ymax = ymin - 1, ymax + 1
    px, py = (xmax - xmin) * 0.05, (ymax - ymin) * 0.06
    xmin, xmax, ymin, ymax = xmin - px, xmax + px, ymin - py, ymax + py

    L, R, B, T = 40, 8, 28, 16
    pw, ph = W - L - R, H - B - T
    sx = lambda x: L + (x - xmin) / (xmax - xmin) * pw      # noqa: E731
    sy = lambda y: B + (y - ymin) / (ymax - ymin) * ph      # noqa: E731

    dr = Drawing(W, H)
    dr.add(String(L, H - 11, _txt(p.get("title", "")), fontName=bold, fontSize=8.5, fillColor=HEAD))
    xt = _log_ticks(xmin, xmax) if logx else _nice_ticks(xmin, xmax)
    yt = _log_ticks(ymin, ymax) if logy else _nice_ticks(ymin, ymax)
    for v in xt:
        x = sx(math.log10(v) if logx else v)
        dr.add(Line(x, B, x, B + ph, strokeColor=colors.HexColor("#e7ebf3"), strokeWidth=0.5))
        dr.add(String(x, B - 9, _fmt_tick(v), fontName=regular, fontSize=6.3, fillColor=MUTED, textAnchor="middle"))
    for v in yt:
        y = sy(math.log10(v) if logy else v)
        dr.add(Line(L, y, L + pw, y, strokeColor=colors.HexColor("#e7ebf3"), strokeWidth=0.5))
        dr.add(String(L - 3, y - 2.2, _fmt_tick(v), fontName=regular, fontSize=6.3, fillColor=MUTED, textAnchor="end"))
    dr.add(Rect(L, B, pw, ph, fillColor=None, strokeColor=colors.HexColor("#8a93a6"), strokeWidth=0.7))
    dr.add(String(L + pw / 2, 4, _txt(p.get("x", "")), fontName=regular, fontSize=7.2, fillColor=INK,
                  textAnchor="middle"))
    ylab = String(0, 0, _txt(p.get("y", "")), fontName=regular, fontSize=7.2, fillColor=INK, textAnchor="middle")
    g = Group(ylab)
    g.transform = (0, 1, -1, 0, 8, B + ph / 2)        # rotated 90° up the left side
    dr.add(g)

    if len(dome) > 1:
        dr.add(PolyLine([v for q in dome for v in (sx(q[0]), sy(q[1]))], strokeColor=DOME_COLOR, strokeWidth=0.8))
    for kind, pts in paths:
        if len(pts) > 1:
            dr.add(PolyLine([v for q in pts for v in (sx(q[0]), sy(q[1]))],
                            strokeColor=KIND_COLOR.get(kind, PATH_COLOR), strokeWidth=1.4, strokeLineJoin=1))

    # states: a dot each; numbers drawn within a few points of each other share one label
    clusters = []
    for name, (x, y) in sorted(states.items(), key=lambda kv: (len(kv[0]), kv[0])):
        X, Y = sx(x), sy(y)
        dr.add(Circle(X, Y, 1.9, fillColor=INK, strokeColor=colors.white, strokeWidth=0.4))
        c = next((c for c in clusters if math.hypot(c[0] - X, c[1] - Y) < 7), None)
        if c:
            c[2].append(name)
        else:
            clusters.append([X, Y, [name]])
    for X, Y, names in clusters:
        anchor, dx = ("end", -3) if X > L + pw - 18 else ("start", 3)
        dy = 3 if Y < B + ph - 10 else -8
        dr.add(String(X + dx, Y + dy, ", ".join(names), fontName=bold, fontSize=6.5, fillColor=INK, textAnchor=anchor))
    return dr


def _plots(sec, st):
    drawings = [d for d in (_plot(p) for p in sec.get("plots") or []) if d is not None]
    if not drawings:
        return []
    rows = [drawings[i:i + 2] + [""] * (2 - len(drawings[i:i + 2])) for i in range(0, len(drawings), 2)]
    t = Table(rows, colWidths=[FRAME_W / 2] * 2, hAlign="LEFT")
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    kinds = {q.get("kind") for p in sec.get("plots") or [] for q in p.get("paths") or []}
    legend = " &nbsp; ".join(f'<font color="{KIND_COLOR[k].hexval().replace("0x", "#")}">━</font> {KIND_TEXT[k]}'
                             for k in ("Qin", "Qout", "Win", "Wout") if k in kinds)
    out = [t]
    if legend:
        out.append(Paragraph(legend + ' &nbsp; <font color="#9aa3b2">━</font> saturation dome', st["small"]))
    return out


# ---------- the document ----------

def _section(sec, st):
    kind = sec.get("kind")
    if kind == "stats":
        body = _stats(sec.get("items") or [], st)
    elif kind == "table":
        body = _table(sec, st)
    elif kind == "facts":
        body = _facts(sec.get("items") or [], st)
    elif kind == "notes":
        body = _notes(sec.get("lines") or [], st)
    elif kind == "sketch":
        dr = _sketch(sec)
        body = [dr] if dr else []
    elif kind == "plots":
        body = _plots(sec, st)
    elif kind == "work":
        body = _work(sec, st)
    else:
        return []
    if not body:
        return []
    head = _heading(sec.get("title") or "", st) if sec.get("title") else [Spacer(1, 6)]
    # keep a short part on one page with its heading
    if kind in ("stats", "facts", "sketch", "plots") or (kind == "table" and len(sec.get("rows") or []) <= 12):
        return [KeepTogether(head + body)]
    return head + body


def build_report(payload):
    """The PDF as bytes."""
    if not isinstance(payload, dict):
        raise ReportError("Nothing to report.")
    st = _styles()
    title = str(payload.get("title") or "TurboSim report")
    version = str(payload.get("version") or "")
    stamp = time.strftime("%B %d, %Y  %H:%M").replace(" 0", " ")

    story = [Paragraph(_esc(title), st["title"]), Spacer(1, 2),
             Paragraph(_esc(f"TurboSim {version} · {stamp}".strip()), st["small"]), Spacer(1, 8)]
    story += _facts(payload.get("facts") or [], st)
    for sec in payload.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        try:
            story += _section(sec, st)
        except Exception as e:      # one broken part must not cost the whole report
            story += [Paragraph(_esc(f"{sec.get('title') or 'A section'} could not be drawn ({e})."), st["small"])]
    if len(story) <= 5:
        story.append(Paragraph("Nothing to report yet. Build or solve a cycle first.", st["base"]))

    def frame(canvas, doc):
        canvas.saveState()
        canvas.setFont(fonts()[0], 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 0.4 * inch, _txt(f"{title} · TurboSim {version}"))
        canvas.drawRightString(PAGE[0] - MARGIN, 0.4 * inch, f"Page {doc.page}")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 0.52 * inch, PAGE[0] - MARGIN, 0.52 * inch)
        canvas.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=PAGE, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                            bottomMargin=0.75 * inch, title=_txt(title), author="TurboSim",
                            subject="Thermodynamic cycle report", creator=f"TurboSim {version}")
    doc.build(story, onFirstPage=frame, onLaterPages=frame)
    return buf.getvalue()


def content_disposition(name):
    """The download header: a plain ASCII name for old clients, the real one (UTF-8) for the rest."""
    from urllib.parse import quote
    plain = name.encode("ascii", "ignore").decode() or "TurboSim report.pdf"
    return f'attachment; filename="{plain}"; filename*=UTF-8\'\'{quote(name)}'


def report_name(title):
    """A file name for the report: the diagram's name, or the date."""
    base = re.sub(r"\.json$", "", str(title or ""), flags=re.I)
    base = re.sub(r'[\\/:*?"<>|]+', "", base).strip() or time.strftime("TurboSim report %Y-%m-%d")
    return f"{base}.pdf"
