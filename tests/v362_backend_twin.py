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


class FakeCore:
    def __init__(self, twin, log):
        self.tw = twin
        self.log = log
        self.lock = threading.RLock()
        self.moves = []

    def emit(self, s):
        self.log.append(str(s))

    def current_tcp_pose(self):
        return self.tw.tcp_pose()

    def object_pose(self, name):
        p, qw = self.tw.cup_pose(name)
        return p, [qw[1], qw[2], qw[3], qw[0]]                  # TF quaternion order xyzw

    def _cartesian_waypoint_plan(self, waypoints_xyz, target_quat, label="", max_step=0.006, max_speed=0.07):
        return {"kind": "line", "xyz": list(waypoints_xyz[-1]), "quat": list(target_quat), "speed": float(max_speed),
                "label": label}

    def _execute_trajectory(self, traj):
        with self.lock:
            self.moves.append((traj["kind"], traj["label"], round(traj["speed"], 4)))
            r = self.tw._move(traj["label"], traj["xyz"], traj["quat"], traj["speed"])
            if not r["ok"]:
                raise RuntimeError(f"twin move missed its target by {r['tcp_error_mm']} mm")
            return True


def pose_plan(core):
    def f(label, xyz, quat, vel=0.12, acc=0.12):
        return {"kind": "pose", "xyz": list(xyz), "quat": list(quat), "speed": 0.10, "label": label}
    return f


def slow(traj, factor=2.0):
    traj["speed"] = traj["speed"] / float(factor)


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
        else:
            tool = last.get("tool")
            names = [s[0] for s in seq]
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


def run_backend_episode(task="Stack cup_1 into cup_2", source="cup_1", target="cup_2", knobs=None):
    """gui_run_task exactly as the backend A2 block calls it -> dict(log, final state, llm, core, twin)"""
    import NERO_V2_7_1_Agent as AG
    log = []
    sh = make_shared()
    tw = SyncedTwin(sh, log=log.append)
    core = FakeCore(tw, log)
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
            "pose_source_mode": lambda: "truth", "shared": sh,
            "time_fn": lambda: tw.t, "sleep_fn": lambda s: tw.step(int(max(0.0, s) / 0.001)),
        })
    finally:
        AG._v360_append_score = orig_append
        if knobs is not None:
            CK.cup_knobs = orig
    a, b = tw.d.body(CK.CUPS[source]), tw.d.body(CK.CUPS[target])
    import NERO_V343_Scenes as SC
    st = SC.cup_task_state(a.xpos, a.xquat, b.xpos, b.xquat, origin="rim")
    return {"log": log, "state": st, "llm": llm, "core": core, "twin": tw, "started": started, "calls": calls,
            "shared": sh, "scores": scores}
