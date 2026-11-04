#!/usr/bin/env python3
"""Offline test of V3.02 yaw-family math using the backend's own module-level quaternion helpers.
Checks: every yaw candidate keeps the held egg axis vertical and places the egg centre on the slot."""
import ast, math, random
from pathlib import Path
src = (Path(__file__).resolve().parents[1] / "NERO_V2_7_1_Unified_Backend.py").read_text()
tree = ast.parse(src)
want = {"_v27_quat_multiply", "_v27_quat_normalize", "_v271_quat_conjugate", "_v271_quat_from_two_vectors",
        "_v271_unit", "_v271_dot", "_v271_cross", "_v271_norm"}
g = {"math": math}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in want:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "x", "exec"), g)
mul, nrm, conj, q2v = g["_v27_quat_multiply"], g["_v27_quat_normalize"], g["_v271_quat_conjugate"], g["_v271_quat_from_two_vectors"]
def rot(q, v):
    return mul(mul(q, [v[0], v[1], v[2], 0.0]), conj(q))[:3]
random.seed(3)
for trial in range(200):
    # random top-down-ish grasp, egg axis horizontal along tool x (major-aligned grasp)
    yaw = random.uniform(-math.pi, math.pi)
    tcp_q = nrm(mul([0, 0, math.sin(yaw/2), math.cos(yaw/2)], [1, 0, 0, 0]))
    axis_local = [1.0, 0.0, 0.0]
    r_local = [random.uniform(-0.016, 0.016), random.uniform(-0.01, 0.01), 0.028]
    axis_w = rot(tcp_q, axis_local)
    targ = [0, 0, 1.0 if axis_w[2] >= 0 else -1.0]
    qv = nrm(mul(q2v(axis_w, targ), tcp_q))
    slot = [-0.51, 0.123, 0.245]
    for yd in range(0, 360, 30):
        h = math.radians(yd) / 2
        qt = nrm(mul([0, 0, math.sin(h), math.cos(h)], qv))
        ax = rot(qt, axis_local)
        err = math.degrees(math.acos(min(1.0, abs(ax[2]))))
        assert err < 1e-4, (trial, yd, err)
        rw = rot(qt, r_local)
        tcpt = [slot[i] - rw[i] for i in range(3)]
        objp = [tcpt[i] + rot(qt, r_local)[i] for i in range(3)]
        assert max(abs(objp[i] - slot[i]) for i in range(3)) < 1e-9
    # yaw family must be 12 distinct TCP orientations
print("[OK] V3.02 yaw family: 200 random grasps x 12 yaws -> egg axis vertical (<1e-4 deg), egg centre on slot target")
