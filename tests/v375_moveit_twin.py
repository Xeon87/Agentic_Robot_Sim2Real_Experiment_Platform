"""V3.75 test helper: the V3.72 sort backend path (v372_backend_twin) with a MoveIt EMULATION in the fake core.

MoveIt in the real simulation carries the OFFICIAL AgileX gripper model (start_nero_sim.sh effector_type:=agx_gripper)
while MuJoCo carries the SO-101.  MoveItSortCore reproduces that: its state validity uses the estimated official
gripper envelope (NERO_V375_Collision_Check.official_gripper_parts) + the wrist links vs the planning-scene table and
bins; a Cartesian path with a colliding point is refused ("Cartesian corridor only xx% collision-free") and a joint
goal in collision fails ("JOINT-GOAL PLANNING FAILED: FAILURE (99999)") - the messages of the user's V3.74 run."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import numpy as np                               # noqa: E402
import NERO_V375_Collision_Check as CC           # noqa: E402
import v362_backend_twin as HB                   # noqa: E402
import v372_backend_twin as SB                   # noqa: E402


class MoveItSortCore(SB.SortCore):
    calls = 0

    def _opening(self):
        return float(self.sh["gripper_actual_mm"].value)

    def v375_state_validity(self, q):
        MoveItSortCore.calls += 1
        parts = CC.official_gripper_parts(q, self._opening()) + CC.so101_parts(q, self._opening())[-3:]
        pairs = CC.check_parts(parts, CC.world_boxes("sort_items"), 0.0)
        return {"available": True, "valid": not pairs, "pairs": pairs, "why": "collision" if pairs else ""}

    def _cartesian_waypoint_plan(self, waypoints_xyz, target_quat, label="", max_step=0.006, max_speed=0.07):
        traj = super()._cartesian_waypoint_plan(waypoints_xyz, target_quat, label, max_step, max_speed)
        qs = [pt.positions for pt in traj.joint_trajectory.points]
        for i, q in enumerate(qs):
            if not self.v375_state_validity(q)["valid"]:
                raise RuntimeError(f"Cartesian corridor only {100.0 * i / max(1, len(qs)):.1f}% collision-free")
        return traj

    def _plan_joint_solution(self, q, velocity_scale=0.1, accel_scale=0.1, planning_attempts=1, planning_time=1.0):
        if not self.v375_state_validity(q)["valid"]:
            raise RuntimeError("JOINT-GOAL PLANNING FAILED: FAILURE (99999)")
        return super()._plan_joint_solution(q, velocity_scale, accel_scale, planning_attempts, planning_time)

    def _make_incremental_joint_trajectory(self, arm_points, xyz_points):
        d = sum(float(np.linalg.norm(np.asarray(a) - np.asarray(b))) for a, b in zip(xyz_points[1:], xyz_points[:-1]))
        self.own_paths = getattr(self, "own_paths", 0) + 1
        return HB.FakeTraj("own", [list(q) for q in arm_points[1:]], max(0.3, d / 0.04), "own path")


def run_backend_episode(**kw):
    orig = SB.SortCore
    SB.SortCore = MoveItSortCore
    try:
        return SB.run_backend_episode(**kw)
    finally:
        SB.SortCore = orig
