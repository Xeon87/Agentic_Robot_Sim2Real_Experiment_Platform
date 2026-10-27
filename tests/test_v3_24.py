#!/usr/bin/env python3
"""V3.24: wider top view behind the base + park pose D (待断电位置).

P1 park pose: default D = the user's recorded angles; save / load round trip; D clamped into the soft limits
   (J2, J4, J7 were moved by hand beyond the URDF limits)
P2 editor view: top view reaches 70 cm behind the base (was 25 cm), side view 70 cm; the "+x zone" warning only for
   zones that lie completely behind the base
P3 wiring: 「待断电位置⚠」 next to 「回 Home⚠」, joint panel go_park / 「当前设为待断电位置」, refused while teaching
"""
import os, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V314_Joint_Control as JC  # noqa: E402
import NERO_V313_Real_Robot as RR  # noqa: E402

assert RR.VERSION.startswith(("V3.2", "V3.3", "V3.4"))
assert JC.PARK_DEFAULT_DEG == [0.1, -100.6, 1.2, 126.0, -0.1, -0.2, 97.3]
tmp = Path(tempfile.mkdtemp(prefix="v324_")) / "park.json"
d = JC.load_park(str(tmp))
assert d["deg"] == JC.PARK_DEFAULT_DEG and "默认" in d["source"]
JC.save_park([1, -90, 2, 110, 0, 0, 80], str(tmp))
assert JC.load_park(str(tmp))["deg"] == [1, -90, 2, 110, 0, 0, 80]
LIM = JC.build_model((ROOT / "config" / "nero.urdf").read_text(), None, None)["limits"]
t, notes = JC.park_target(JC.PARK_DEFAULT_DEG, LIM)
assert [n[0] for n in notes] == [1, 3, 6], notes
for j, v in enumerate(t):
    lo, hi = JC.soft_range_deg(*LIM[j])
    assert lo - 1e-6 <= v <= hi + 1e-6
assert abs(t[1] - JC.soft_range_deg(*LIM[1])[0]) < 0.01 and abs(t[6] - JC.soft_range_deg(*LIM[6])[1]) < 0.01
print(f"[P1] D = {JC.PARK_DEFAULT_DEG}; MoveIt target {t} (J2/J4/J7 clamped to the soft limits); save/load ok")

E = (ROOT / "NERO_V316_Safety_Editor.py").read_text()
assert "TOP_X = (-0.95, 0.70)" in E and 'return (-0.45, 1.00) if self.v_side.get() == "XZ"' in E
assert 'all(p[0] > -0.05 for p in self.fence()["polygon"])' in E
print("[P2] top view: 95 cm forward .. 70 cm behind; side view 45 cm behind; +x warning only for zones fully behind")

C = (ROOT / "NERO_V312_Console.py").read_text()
J = (ROOT / "NERO_V314_Joint_Control.py").read_text()
i_home, i_park = C.index('("回 Home⚠", "move_home"'), C.index('("待断电位置⚠", "park"')
assert 0 < i_park - i_home < 120 and "jpanel.go_park()" in C
assert "def go_park(" in J and "def on_park_record(" in J and "当前设为待断电位置" in J and "def park_zone_hint(" in J
assert J.index('[WARN] [V3.24 park] NERO 在示教模式') < J.index('def on_park_record(')
print("[P3] 「待断电位置⚠」 next to 「回 Home⚠」 -> jpanel.go_park(); 「当前设为待断电位置」; refused while teaching")
print("ALL V3.24 TESTS PASS")
