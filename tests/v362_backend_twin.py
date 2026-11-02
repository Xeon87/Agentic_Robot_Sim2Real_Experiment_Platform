"""V3.62 test helper: a fake simulation backend for NERO_V362_Cup_Skills.gui_run_task.

The real BackendIO adapter is used unchanged; underneath it:
  * FakeCore    = skill_core stand-in (current_tcp_pose / object_pose / _cartesian_waypoint_plan / _execute_trajectory /
                  emit) whose motions run in the headless MuJoCo twin (TwinIO),
  * Shared      = the backend's multiprocessing Values (gripper / hold / contact / safety), synchronised with the twin's
                  gripper before every physics step (the twin runs the backend's own gripper logic),
  * a simulation clock (time_fn / sleep_fn) so every wait in BackendIO is measured in simulated seconds,
  * ScriptedLLM = replaces NeroLLMAgent._post_responses: a deterministic "model" that follows the CS skill contract
                  (it reads recommended_next_actions), so the real agent transport / loop / scoring code runs.
No ROS, no network.
"""
import json
import os
import sys
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
import NERO_V362_Cup_Skills as CK  # noqa: E402


class V:
    def __init__(self, v):
        self.value = v


def make_shared():
    return {"gripper_target_mm": V(20.0), "gripper_actual_mm": V(20.0), "gripper_force_mode": V(False),
            "gripper_force_N": V(5.0), "clamp_speed_mult_shared": V(1.0), "hold_enabled_shared": V(True),
            "hold_off_request": V(False), "hold_active_shared": V(False), "contact_left_n_shared": V(0.0),
            "contact_right_n_shared": V(0.0), "safety_trip_count": V(0)}


class SyncedTwin(CK.TwinIO):
    """TwinIO whose gripper follows the backend shared values (and reports back into them)."""

    def __init__(self, shared, **kw):
        self.sh = shared
        super().__init__(**kw)

    def step(self, n=1):
        for _ in range(int(n)):
            sh = getattr(self, "sh", None)
            if sh is not None:
                if sh["hold_off_request"].value:
                    sh["hold_off_request"].value = False
                    self._release_hold()
                self.force_mode = bool(sh["gripper_force_mode"].value)
                self.force_n = float(sh["gripper_force_N"].value)
                self.target_q = max(0.0, min(0.05, float(sh["gripper_target_mm"].value) / 2000.0))
            super().step(1)
            if sh is not None:
                sh["gripper_actual_mm"].value = self.opening_mm()
                sh["hold_active_shared"].value = self.hold is not None
                sh["contact_left_n_shared"].value = float(getattr(self, "L", 0.0))
                sh["contact_right_n_shared"].value = float(getattr(self, "Rn", 0.0))


class _Pt:
    def __init__(self, q):
        self.positions = list(q)


class _JT:
    def __init__(self, qs):
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.points = [_Pt(q) for q in qs]


class FakeTraj:
    """MoveIt RobotTrajectory stand-in: joint_trajectory.joint_names / points[].positions (+ timing info)"""

    def __init__(self, kind, qs, dur, label):
        self.kind, self.label, self.dur = kind, label, float(dur)
        self.joint_trajectory = _JT(qs)


class FakeCore:
    """skill_core stand-in on the twin (V3.63: joint trajectories like MoveIt; optional injected safety trip)"""

    def __init__(self, twin, log, shared, trip_on=None, trips=1):
        self.tw = twin
        self.log = log
        self.sh = shared
        self.lock = threading.RLock()
        self.moves = []
        self.trip_on = trip_on                    # substring of a motion label -> simulated safety trip(s) there
        self.trips_left = int(trips)
        self.latched = False

    def emit(self, s):
        self.log.append(str(s))

    def current_tcp_pose(self):
        return self.tw.tcp_pose()

    def object_pose(self, name):
        p, qw = self.tw.cup_pose(name)
        return p, [qw[1], qw[2], qw[3], qw[0]]                  # TF quaternion order xyzw

    def _current_joint_state_copy(self):
        return list(self.tw.q)

    @staticmethod
    def _arm_positions_from_joint_state(js):
        return list(js)

    def _cartesian_waypoint_plan(self, waypoints_xyz, target_quat, label="", max_step=0.006, max_speed=0.07):
        qs, why = self.tw._cartesian_q_path(list(waypoints_xyz[-1]), list(target_quat))
        if qs is None:
            raise RuntimeError(why)
        T0 = self.tw.arm.fk(np.asarray(qs[0], float))
        dist = float(np.linalg.norm(np.asarray(waypoints_xyz[-1]) - T0[:3, 3]))
        return FakeTraj("line", qs[1:], max(0.2, dist / max(max_speed, 1e-3)), label)

    def _plan_joint_solution(self, q, velocity_scale=0.1, accel_scale=0.1, planning_attempts=1, planning_time=1.0):
        q0 = np.asarray(self.tw.q, float)
        qs = [list(q0 + u * (np.asarray(q, float) - q0)) for u in np.linspace(0.05, 1.0, 20)]
        dq = float(np.max(np.abs(np.asarray(q) - q0)))
        return FakeTraj("joints", qs, max(0.3, dq / 0.35), "joint goal")

    def _execute_trajectory(self, traj):
        with self.lock:
            if self.latched:
                raise RuntimeError("PHYSICAL SAFETY STOP: latch held")
            self.moves.append((traj.kind, self.cur_label, round(traj.dur, 2)))
            qs = [list(self.tw.q)] + [list(pt.positions) for pt in traj.joint_trajectory.points]
            if self.trip_on and self.trips_left > 0 and self.trip_on in (self.cur_label or ""):
                self.trips_left -= 1                      # run half the path, freeze, count a trip, latch
                self.tw._run_q_path(qs[: max(2, len(qs) // 2)], traj.dur / 2)
                self.sh["safety_trip_count"].value += 1
                self.latched = True
                raise RuntimeError("PHYSICAL SAFETY STOP: injected trip (test)")
            self.tw._run_q_path(qs, traj.dur)
            return True


def pose_plan(core):
    def f(label, xyz, quat, vel=0.12, acc=0.12):
        q = CK.ik_from(core.tw.q, xyz, CK.quat_xyzw_to_R(quat))
        if q is None:
            raise RuntimeError("cuRobo: no IK")
        return core._plan_joint_solution(q)
    return f


def slow(traj, factor=2.0):
    traj.dur = traj.dur * float(factor)


class ScriptedLLM:
    """deterministic stand-in for the model API: follows the CS contract from the tool results."""

    def __init__(self, source="cup_1", target="cup_2"):
        self.src, self.dst = source, target
        self.n = 0
        self.requests = []

    def __call__(self, payload, api_key, timeout_s=180.0):
        self.requests.append(payload)
        self.n += 1
        last = None
        if isinstance(payload.get("input"), list) and payload["input"]:
            try:
                last = json.loads(payload["input"][-1]["output"])
            except Exception:
                last = None
        seq = [("observe_scene", {}), ("plan_cup_stack", {"source_cup": self.src, "target_cup": self.dst}),
               ("pick_cup", {"source_cup": self.src}), ("place_cup_into", {"target_cup": self.dst}),
               ("verify_stack", {"upper_cup": self.src, "lower_cup": self.dst})]
        if last is None:
            name, args = seq[0]
        elif getattr(self, "retry_after_observe", None) and last.get("tool") == "observe_scene":
            name, args = self.retry_after_observe
            self.retry_after_observe = None
        else:
            tool = last.get("tool")
            names = [s[0] for s in seq]
            nxt = last.get("recommended_next_actions") or []
            if (not last.get("ok") and tool in nxt and "observe_scene" in nxt and not getattr(self, "retried", False)):
                self.retried = True                       # the contract allows ONE retry (after observe_scene)
                self.retry_after_observe = (tool, dict(seq)[tool])
                return {"id": f"r{self.n}", "output": [{"type": "function_call", "name": "observe_scene",
                                                         "call_id": f"c{self.n}", "arguments": "{}"}],
                        "usage": {"input_tokens": 100, "output_tokens": 20}}
            if tool == "verify_stack" or not last.get("ok"):
                ok = bool(last.get("ok"))
                return {"id": f"r{self.n}", "output": [{"type": "message", "content": [{"type": "output_text", "text":
                        (f"{self.src} is stacked inside {self.dst} (verified)." if ok else
                         f"Stopped: {tool} failed ({last.get('reason')}).")}]}],
                        "usage": {"input_tokens": 100, "output_tokens": 20}}
            name, args = seq[names.index(tool) + 1]
        return {"id": f"r{self.n}", "output": [{"type": "function_call", "name": name, "call_id": f"c{self.n}",
                                                 "arguments": json.dumps(args)}],
                "usage": {"input_tokens": 100, "output_tokens": 20}}


def run_backend_episode(task="Stack cup_1 into cup_2", source="cup_1", target="cup_2", knobs=None, trip_on=None,
                        trips=1, pose_mode="truth", wrist=False, camera_up=True):
    """gui_run_task exactly as the backend A2 block calls it -> dict(log, final state, llm, core, twin)"""
    import NERO_V2_7_1_Agent as AG
    log = []
    sh = make_shared()
    tw = SyncedTwin(sh, log=log.append)
    core = FakeCore(tw, log, sh, trip_on=trip_on, trips=trips)
    core.cur_label = ""
    agent = AG.NeroLLMAgent(emit=log.append, tool_runner=None)
    llm = ScriptedLLM(source, target)
    agent._post_responses = llm
    started = {}

    def background(label, fn, lock_motion=False):
        started["label"] = label
        try:
            started["result"] = fn()
        except Exception as exc:
            started["error"] = str(exc)

    cfg = {"provider": "openai", "model": "gpt-test", "api_key": "sk-test", "reasoning_effort": "low",
           "max_tool_steps": 60, "dry_run": False, "task": task}
    calls = {"start_check": 0, "go_q0": 0}

    def start_check(edge):
        calls["start_check"] += 1
        return True

    def go_q0(label):
        calls["go_q0"] += 1
        tw.go_home(label)

    scores = []
    orig_append = AG._v360_append_score
    AG._v360_append_score = lambda sc, task, model: scores.append(dict(sc, task=task, model=model))   # no logs/ writes
    def _clear(label):
        calls["clear_latch"] = calls.get("clear_latch", 0) + 1
        core.latched = False
        return True, "ok"

    # BackendIO labels each motion; remember it so the fake core can inject a trip on a named motion
    _orig_ml, _orig_mj = CK.BackendIO.move_line, CK.BackendIO.move_joints

    def _ml(self, label, *a, **k):
        core.cur_label = label
        return _orig_ml(self, label, *a, **k)

    def _mj(self, label, *a, **k):
        core.cur_label = label
        return _orig_mj(self, label, *a, **k)
    CK.BackendIO.move_line, CK.BackendIO.move_joints = _ml, _mj
    if knobs is not None:
        orig = CK.cup_knobs
        CK.cup_knobs = lambda profile=None: (dict(CK.CUP_KNOB_DEFAULTS, **knobs), {"code": "CS", "short_name": "Cup Stack",
                                                                                     "skill_id": "cup_stack_rim_pinch",
                                                                                     "source": "test"})
    try:
        CK.gui_run_task({
            "skill_core": core, "agent": agent, "cfg_fn": lambda: dict(cfg), "background": background,
            "log": log.append, "messagebox": None, "require_arm": lambda: True, "pose_plan": pose_plan(core),
            "slow": slow, "go_q0": go_q0, "start_check": start_check, "safety_gate": lambda: True,
            "pose_source_mode": lambda: pose_mode, "shared": sh,
            "wrist_selected": lambda: wrist, "camera_available": lambda: camera_up,
            "capture": lambda tag: (calls.__setitem__("captures", calls.get("captures", 0) + 1)
                                    or {"camera": "fixed_d435i", "tag": tag, "cups": tw.perceive(tag)}),
            "time_fn": lambda: tw.t, "sleep_fn": lambda s: tw.step(int(max(0.0, s) / 0.001)),
            "latch_pending": lambda: core.latched, "clear_latch": _clear,
        })
    finally:
        CK.BackendIO.move_line, CK.BackendIO.move_joints = _orig_ml, _orig_mj
        AG._v360_append_score = orig_append
        if knobs is not None:
            CK.cup_knobs = orig
    a, b = tw.d.body(CK.CUPS[source]), tw.d.body(CK.CUPS[target])
    import NERO_V343_Scenes as SC
    st = SC.cup_task_state(a.xpos, a.xquat, b.xpos, b.xquat, origin="rim")
    return {"log": log, "state": st, "llm": llm, "core": core, "twin": tw, "started": started, "calls": calls,
            "shared": sh, "scores": scores}
