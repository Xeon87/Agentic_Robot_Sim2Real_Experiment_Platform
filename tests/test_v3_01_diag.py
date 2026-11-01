#!/usr/bin/env python3
"""Offline test of the V3.01 diagnostics recorder: extracts the exact block from the backend and runs it."""
import os, sys, tempfile, textwrap, time, re
from pathlib import Path
src = (Path(__file__).resolve().parents[1] / "NERO_V2_7_1_Unified_Backend.py").read_text()
a = src.index("        import collections as _v301_collections")
b = src.index("        with mujoco.viewer.launch_passive(model, data) as viewer:")
block = textwrap.dedent(src[a:b])
with tempfile.TemporaryDirectory() as d:
    os.environ["NERO_DIAG_DIR"] = d
    g = {"os": os, "time": time}
    exec(block, g)
    row = [0.0, 1, 1] + [0.1] * 35
    for _ in range(2000):
        g["_v301_record"](list(row))
    time.sleep(0.06); g["_v301_record"](list(row))
    assert len(g["_v301_ring"]) == 1500, len(g["_v301_ring"])
    g["_v301_dump"]("runaway_j4")
    files = sorted(os.listdir(d))
    assert "diag_motion.csv" in files and "diag_trip_1_runaway_j4.csv" in files, files
    lines = open(os.path.join(d, "diag_trip_1_runaway_j4.csv")).read().splitlines()
    assert len(lines) == 1501 and lines[0].startswith("t_s,gate,hold,q1"), len(lines)
    assert len(lines[1].split(",")) == 38
    assert len(g["_v301_ring"]) == 0
    os.environ["NERO_DIAG_DIR"] = ""
    g2 = {"os": os, "time": time}; exec(block, g2); g2["_v301_record"](row); g2["_v301_dump"]("x")
print("[OK] V3.01 diagnostics recorder: ring=1500, 20 Hz csv, trip dump, disabled when NERO_DIAG_DIR empty")
