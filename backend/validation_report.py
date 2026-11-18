"""Printable PDF of the textbook validation (validate_textbook.py --pdf FILE).

One page of summary, then one block per worked example: the book, the
example number, the given data, and each answer from the book next to
TurboSim's with the percent difference and the tolerance it was held to.
"""
import datetime
import pathlib
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from report import fonts

BOOKS = {
    "Moran": "M. J. Moran, H. N. Shapiro, D. D. Boettner, M. B. Bailey, <i>Fundamentals of Engineering "
             "Thermodynamics</i>, 9th ed., Wiley.",
    "Penoncello": "S. G. Penoncello, <i>Thermal Energy Systems: Design and Analysis</i>, 2nd ed., CRC Press.",
    "Dick": "E. Dick, <i>Fundamentals of Turbomachines</i>, Springer, 2015.",
}

INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#52606d")
RULE = colors.HexColor("#cbd2d9")
HEAD = colors.HexColor("#e4e7eb")
PASS = colors.HexColor("#1b7f3b")
FAIL = colors.HexColor("#b42318")


def app_version():
    p = pathlib.Path(__file__).resolve().parent.parent / "frontend" / "src" / "version.js"
    try:
        m = re.search(r"VERSION\s*=\s*['\"]([^'\"]+)", p.read_text(encoding="utf-8"))
        return m.group(1) if m else "?"
    except OSError:
        return "?"


def _num(x):
    if x is None:
        return "-"
    a = abs(x)
    if a >= 1e5:
        exp = int(f"{x:e}".split("e")[1])
        sup = str(exp).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
        return f"{x / 10 ** exp:.4g} × 10{sup}"
    if a >= 1000:
        return f"{x:,.1f}"
    if a >= 100:
        return f"{x:.1f}"
    if a >= 10:
        return f"{x:.2f}"
    if a >= 1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def _err(c):
    if not c.book:
        return "-"
    return f"{(c.app - c.book) / abs(c.book) * 100:+.2f} %"


def _tol(c):
    if c.abs_tol is not None:
        return f"± {_num(c.abs_tol)} {c.unit}".strip()
    return f"± {c.rel * 100:g} %"


def write_pdf(path, rows, given, project=None):
    regular, bold = fonts()
    ss = {
        "title": ParagraphStyle("title", fontName=bold, fontSize=20, leading=24, textColor=INK, spaceAfter=4),
        "sub": ParagraphStyle("sub", fontName=regular, fontSize=10.5, leading=14, textColor=MUTED, spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName=bold, fontSize=13, leading=16, textColor=INK, spaceBefore=10, spaceAfter=4),
        "h3": ParagraphStyle("h3", fontName=bold, fontSize=10.5, leading=13, textColor=INK, spaceBefore=4),
        "body": ParagraphStyle("body", fontName=regular, fontSize=9.5, leading=13, textColor=INK, alignment=TA_LEFT),
        "small": ParagraphStyle("small", fontName=regular, fontSize=8.5, leading=11, textColor=MUTED),
        "cell": ParagraphStyle("cell", fontName=regular, fontSize=8.5, leading=10.5, textColor=INK),
        "cellb": ParagraphStyle("cellb", fontName=bold, fontSize=8.5, leading=10.5, textColor=INK),
    }
    version = app_version()
    pname = (project or {}).get("name")
    named = list(zip(rows, given["order"]))
    part1 = [x for x in named if x[0][0] != pname]
    part2 = [x for x in named if x[0][0] == pname]
    rows1 = [r for r, _ in part1]
    n_ex = len(rows1)
    n_q = sum(len(r[3]) for r in rows1)
    n_ok = sum(1 for r in rows1 if r[4] is None and all(c.ok for c in r[3]))
    worst = max((abs(c.app - c.book) / abs(c.book) * 100 for r in rows1 for c in r[3] if c.book), default=0.0)
    n_q2 = sum(len(r[3]) for r, _ in part2)
    ok2 = all(r[4] is None and all(c.ok for c in r[3]) for r, _ in part2)

    sub = (f"TurboSim {version} · {datetime.date.today():%d %B %Y} · Part 1: {n_ok} of {n_ex} textbook examples "
           f"pass ({n_q} answers)")
    if part2:
        sub += f" · Part 2: ME 313 project {'reproduced' if ok2 else 'NOT reproduced'} ({n_q2} values)"
    story = [
        Paragraph("TurboSim validation report", ss["title"]),
        Paragraph(sub, ss["sub"]),
        Paragraph("Part 1: Textbook worked examples", ss["h2"]),
        Paragraph("What was checked", ss["h3"]),
        Paragraph(
            "Each worked example below was set up the way a student would draw it in TurboSim (states, devices "
            "and assumptions) and solved by the same solver that runs behind the app's Solve button, the Exergy "
            "panel and the heat exchanger design tools. No answer from the book is fed back into the solver; "
            "the only inputs are the example's given data. TurboSim's answers are then compared with the "
            "answers printed in the book.", ss["body"]),
        Spacer(1, 5),
        Paragraph(
            "Small differences are expected and are not errors. The books read property tables (steam, R-134a, "
            "the ideal-gas air table, the psychrometric chart) and round intermediate values; TurboSim computes "
            "properties from the CoolProp reference equations of state, or, where the book's model is the ideal "
            "gas, from the same ideal-gas model (constant cp with k = 1.4, or the air table with cp varying with "
            "temperature). Each answer was held to a tolerance of typically 1 to 2 %, tight enough that a wrong "
            "equation, sign or state would fail. English-unit examples were run in SI after converting the given "
            "data; the book's answers are converted the same way.", ss["body"]),
        Spacer(1, 5),
        Paragraph(f"Largest difference from the book over all {n_q} answers: {worst:.2f} %.", ss["body"]),
        Paragraph("Books", ss["h3"]),
    ]
    for key, text in BOOKS.items():
        story.append(Paragraph(f"<b>{key}</b>: {text}", ss["body"]))
    story.append(Paragraph("Summary", ss["h3"]))

    head = ["Book", "Example", "Problem", "Checks", "Largest diff.", "Result"]
    data = [[Paragraph(h, ss["cellb"]) for h in head]]
    for book, ex, title, checks, err in rows1:
        ok = err is None and all(c.ok for c in checks)
        diffs = [abs(c.app - c.book) / abs(c.book) * 100 for c in checks if c.book]
        data.append([
            Paragraph(book, ss["cell"]), Paragraph(ex, ss["cell"]), Paragraph(title, ss["cell"]),
            Paragraph(str(len(checks)), ss["cell"]),
            Paragraph(f"{max(diffs):.2f} %" if diffs else "-", ss["cell"]),
            Paragraph(f"<font color='{PASS.hexval() if ok else FAIL.hexval()}'><b>{'Pass' if ok else 'FAIL'}"
                      f"</b></font>", ss["cell"]),
        ])
    t = Table(data, colWidths=[0.85 * inch, 0.75 * inch, 3.25 * inch, 0.6 * inch, 0.95 * inch, 0.6 * inch],
              repeatRows=1)
    t.setStyle(_grid())
    story += [t, PageBreak(), Paragraph("Part 1: Worked examples", ss["h2"])]
    story += _blocks(part1, given, ss, pname)

    if part2:
        story += [PageBreak(), Paragraph("Part 2: Fact-check of the ME 313 project", ss["h2"]),
                  Paragraph(
                      "Part 2 of the ME 313 project (a nuclear-heated Rankine cycle with three closed feedwater "
                      "heaters, a moisture separator and a wet cooling tower; the 4-device simple cycle of Part 1 "
                      "is left out) was rebuilt in TurboSim from its given data alone, with the project's own "
                      "state numbers, and solved. The original handout was not available, so the givens were "
                      "taken from the project's EES code and report. Every state point and result was then "
                      "compared with the project's EES solution. The diagram is in TurboSim under Templates as "
                      "\"Nuclear Rankine (ME 313)\".", ss["body"]),
                  Paragraph("Given data", ss["h3"])]
        data = [[Paragraph(h, ss["cellb"]) for h in ("Item", "Given")]]
        data += [[Paragraph(a, ss["cell"]), Paragraph(b, ss["cell"])] for a, b in project["givens"]]
        tg = Table(data, colWidths=[1.8 * inch, 5.2 * inch])
        tg.setStyle(_grid())
        story += [tg, Paragraph("The system as built in TurboSim", ss["h3"])]
        data = [[Paragraph(h, ss["cellb"]) for h in ("Device", "States", "Set up as")]]
        data += [[Paragraph(a, ss["cell"]), Paragraph(b, ss["cell"]), Paragraph(c, ss["cell"])]
                 for a, b, c in project["system"]]
        ts = Table(data, colWidths=[1.8 * inch, 1.9 * inch, 3.3 * inch])
        ts.setStyle(_grid())
        story.append(ts)
        img = pathlib.Path(__file__).resolve().parent / "validation_assets" / "me313_diagram.png"
        if img.exists():
            from reportlab.lib.utils import ImageReader
            from reportlab.platypus import Image
            iw, ih = ImageReader(str(img)).getSize()
            w = 6.9 * inch
            story += [Spacer(1, 6), Image(str(img), width=w, height=w * ih / iw),
                      Paragraph("The Part 2 diagram as drawn and solved in TurboSim (File > Publish), with the "
                                "project's state numbers. State 30, the makeup water, enters the mixing chamber's "
                                "second inlet.", ss["small"])]
        story += [Paragraph("Fact-check findings", ss["h3"])]
        for i, note in enumerate(project["notes"], 1):
            story.append(Paragraph(f"{i}. {note}", ss["body"]))
            story.append(Spacer(1, 3))
        story += [PageBreak()] + _blocks(part2, given, ss, pname, ref="EES")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(regular, 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.75 * inch, 0.5 * inch, f"TurboSim {version} validation report")
        canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.8 * inch,
                            title=f"TurboSim {version} validation report", author="TurboSim")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def _blocks(named, given, ss, pname, ref="Textbook"):
    out = []
    for (book, ex, title, checks, err), name in named:
        ok = err is None and all(c.ok for c in checks)
        head = f"{book}, {ex}: {title}" if book == pname else f"{book}, Example {ex}: {title}"
        block = [Paragraph(head, ss["h3"])]
        if given["text"].get(name):
            block.append(Paragraph(f"<b>Given.</b> {given['text'][name]}", ss["body"]))
        block.append(Spacer(1, 3))
        data = [[Paragraph(h, ss["cellb"]) for h in ("Answer", ref, "TurboSim", "Unit", "Diff.",
                                                       "Tolerance", "")]]
        for c in checks:
            fine = c.what.startswith("η at") or c.what == "Best efficiency"    # the sweep is flat: show 6 places
            nf = (lambda x: f"{x:.6f}") if fine else _num
            data.append([Paragraph(c.what, ss["cell"]), Paragraph(nf(c.book), ss["cell"]),
                         Paragraph(nf(c.app), ss["cell"]), Paragraph(c.unit or "-", ss["cell"]),
                         Paragraph(_err(c), ss["cell"]), Paragraph(_tol(c), ss["cell"]),
                         Paragraph("✓" if c.ok else "✗", ss["cell"])])
        if err:
            data.append([Paragraph(f"Could not solve: {err}", ss["cell"])] + [""] * 6)
        tt = Table(data, colWidths=[2.15 * inch, 0.95 * inch, 0.95 * inch, 0.7 * inch, 0.7 * inch, 0.9 * inch,
                                    0.25 * inch], repeatRows=1)
        tt.setStyle(_grid())
        tail = [Spacer(1, 2),
                Paragraph("Result: " + ("all values within tolerance." if ok else "NOT within tolerance."),
                          ss["small"]), Spacer(1, 6)]
        if name == "project_states":
            out += block + [Spacer(1, 3), _state_table(checks, ss)] + tail
            continue
        if len(checks) > 30:
            out += block + [Spacer(1, 3), tt] + tail      # long tables flow across pages
        else:
            out.append(KeepTogether(block + [Spacer(1, 3), tt] + tail))
    return out


def _state_table(checks, ss):
    """The state-point table the way EES prints its Arrays Table: one row per
    state, EES and TurboSim side by side for T, h, s, ṁ and x (or ω)."""
    cols = [("Temperature", "T [°F]"), ("Enthalpy", "h [Btu/lbm]"), ("Entropy", "s [Btu/lbm·R]"),
            ("Mass flow", "ṁ [10⁶ lbm/h]"), ("Quality", "x / ω")]
    by = {}
    for c in checks:
        st, q = c.what.split(": ", 1)
        n = int(st.split()[1])
        by.setdefault(n, {})["Quality" if q in ("Quality", "Humidity ratio") else q] = c
    sm = ParagraphStyle("tiny", parent=ss["cell"], fontSize=7.2, leading=9)
    smb = ParagraphStyle("tinyb", parent=ss["cellb"], fontSize=7.2, leading=9)
    head1 = [Paragraph("State", smb)]
    head2 = [Paragraph("", smb)]
    for _, label in cols:
        head1 += [Paragraph(label, smb), ""]
        head2 += [Paragraph("EES", smb), Paragraph("TurboSim", smb)]
    head1.append(Paragraph("Worst diff.", smb))
    head2.append("")
    data = [head1, head2]
    bad = []
    for n in sorted(by):
        row = [Paragraph(str(n), sm)]
        worst = 0.0
        for key, _ in cols:
            c = by[n].get(key)
            if c is None:
                row += ["", ""]
                continue
            k = 1e-6 if key == "Mass flow" else 1.0
            row += [Paragraph(_num(c.book * k), sm), Paragraph(_num(c.app * k), sm)]
            if c.book:
                worst = max(worst, abs(c.app - c.book) / abs(c.book) * 100)
            if not c.ok:
                bad.append((len(data), c))
        row.append(Paragraph(f"{worst:.2f} %", sm))
        data.append(row)
    w = [0.4 * inch] + [0.58 * inch] * 10 + [0.6 * inch]
    t = Table(data, colWidths=w, repeatRows=2)
    style = _grid()
    style.add("BACKGROUND", (0, 1), (-1, 1), HEAD)
    for i in range(len(cols)):
        style.add("SPAN", (1 + 2 * i, 0), (2 + 2 * i, 0))
    for r, _ in bad:
        style.add("TEXTCOLOR", (0, r), (-1, r), FAIL)
    t.setStyle(style)
    return t


def _grid():
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ])
