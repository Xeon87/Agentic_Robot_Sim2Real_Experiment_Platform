#!/usr/bin/env python3
"""V3.16.2: Tk 8.6 (Ubuntu 22.04) startup segfault guard — static checks on the GUI sources.

T1 no tkinter.font.Font objects are created (a garbage-collected Font = "font delete" -> Tk 8.6/Xft dangling font
   cache -> segfault at the next widget creation; V3.16.1 crashed at startup this way)
T2 no characters that fontconfig may render with the colour-emoji font (Tk 8.6 + Xft cannot draw them safely)
T3 fixed_label measures the line height with "font metrics" and caches it
"""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
GUI = ["NERO_V312_Console.py", "NERO_V314_Joint_Control.py", "NERO_V315_Consistency_Safety.py", "NERO_V316_Safety_Editor.py"]
OK_SYMBOLS = set("✓✗✎⚠")
for f in GUI:
    src = (ROOT / f).read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert not re.search(r"\bFont\(", code), f"{f}: creates a tkinter Font object"
    for i, line in enumerate(src.splitlines(), 1):
        bad = [c for c in line if ord(c) >= 0x1F000 or (0x2600 <= ord(c) <= 0x27BF and c not in OK_SYMBOLS)]
        assert not bad, f"{f}:{i}: emoji-range character {bad}"
print("[T1] no tkinter Font objects in the GUI code")
print("[T2] no colour-emoji characters in the GUI code")
J = (ROOT / "NERO_V314_Joint_Control.py").read_text(encoding="utf-8")
assert '"font", "metrics", font, "-linespace"' in J and "_LINESPACE" in J
C = (ROOT / "NERO_V312_Console.py").read_text(encoding="utf-8")
assert 'VERSION = "V3.' in C
print("[T3] fixed_label: line height via 'font metrics' (cached)")
print("ALL V3.16.2 TESTS PASS")
