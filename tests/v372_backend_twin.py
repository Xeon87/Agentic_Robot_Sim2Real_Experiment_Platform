"""V3.72 test helper: NERO_V372_Sort_Skills.gui_run_task through the real SortBackendIO on the V3.62 fake backend
(v362_backend_twin: FakeCore on the headless twin, backend shared values, simulation clock) - here with the sort_items
bins + items.  ScriptedSortLLM stands in for the model API: it reads observe_scene, assigns every item by its label
the way the prompt "fruit -> red bin, everything else -> blue bin" asks, and follows the tool contract.  No network."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import NERO_V343_Scenes as SC                    # noqa: E402
import NERO_V372_Sort_Skills as S                # noqa: E402
import v362_backend_twin as HB                   # noqa: E402

FRUIT_WORDS = ("banana", "apple", "orange", "pear", "grape")


class SortCore(HB.FakeCore):
    """V3.73: object_pose exactly like the backend (NERO_V71 skill core): TF frame "sim_obj_" + name, published for
    "sim_" + every key of NERO_V343_Scenes.tf_body_map() -> an unknown frame raises the backend's TF error"""
    def object_pose(self, name):
        frame = f"sim_obj_{name}"
        bmap = SC.tf_body_map()
        key = frame[len("sim_"):]
        if key not in bmap:
            raise RuntimeError(f"TF unavailable: base_link -> {frame}: \"{frame}\" passed to lookupTransform argument "
                               "source_frame does not exist. ")
        p, qw = self.tw.object_pose(bmap[key])
        return p, [qw[1], qw[2], qw[3], qw[0]]


class SyncedSortTwin(HB.SyncedTwin):
    def __init__(self, shared, log=None, seed=None):
        self.sh = shared
        S.CK.TwinIO.__init__(self, log=log, gripper="so101", world_xml=SC.sort_world_xml(seed, bare=True),
                             hold_names=SC.SORT_ITEM_NAMES, soft_pads=True)   # V3.76 SOFT PADS

    def object_pose(self, body):
        b = self.d.body(body)
        return [float(v) for v in b.xpos], [float(v) for v in b.xquat]


class ScriptedSortLLM:
    def __init__(self, fruit_box="red", other_box="blue"):
        self.fruit, self.other = fruit_box, other_box
        self.n = 0
        self.queue = None
        self.requests = []

    def _msg(self, text):
        return {"id": f"r{self.n}", "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
                "usage": {"input_tokens": 100, "output_tokens": 20}}

    def _call(self, name, args):
        return {"id": f"r{self.n}", "output": [{"type": "function_call", "name": name, "call_id": f"c{self.n}",
                                                 "arguments": json.dumps(args)}],
                "usage": {"input_tokens": 100, "output_tokens": 20}}

    def __call__(self, payload, api_key, timeout_s=180.0):
        self.requests.append(payload)
        self.n += 1
        last = None
        if isinstance(payload.get("input"), list) and payload["input"]:
            try:
                last = json.loads(payload["input"][-1]["output"])
            except Exception:
                last = None
        if last is None:
            return self._call("observe_scene", {})
        if last.get("tool") == "observe_scene" and self.queue is None:
            self.queue = []
            for it in last["scene"]["items"]:
                box = self.fruit if any(w in it["label"].lower() for w in FRUIT_WORDS) else self.other
                if it["in_bin"] != box:
                    self.queue += [("pick_object", {"item_id": it["id"], "box": box}), ("place_in_box", {"box": box})]
            self.queue += [("check_bins", {}), ("return_to_safe_pose", {})]
        if last.get("tool") == "return_to_safe_pose":
            return self._msg("sorted: fruit -> red bin, other items -> blue bin (check_bins)")
        if not last.get("ok") and last.get("tool") == "pick_object":
            while self.queue and self.queue[0][0] == "place_in_box":
                self.queue.pop(0)                       # skip the place of an item that was not picked
        if not self.queue:
            return self._msg("done")
        name, args = self.queue.pop(0)
        return self._call(name, args)


def run_backend_episode(task="Put the fruit into the red bin and everything else into the blue bin", seed=None,
                        pose_mode="truth"):
    import NERO_V2_7_1_Agent as AG
    log = []
    sh = HB.make_shared()
    tw = SyncedSortTwin(sh, log=log.append, seed=seed)
    env_keep = {k_: os.environ.get(k_) for k_ in ("NERO_SCENE", "NERO_V343_ACTIVE_SCENE")}
    os.environ["NERO_SCENE"], os.environ["NERO_V343_ACTIVE_SCENE"] = "sort_items", "sort_items"
    core = SortCore(tw, log, sh)
    core.cur_label = ""
    agent = AG.NeroLLMAgent(emit=log.append, tool_runner=None)
    llm = ScriptedSortLLM()
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
    scores = []
    orig_append = AG._v360_append_score
    AG._v360_append_score = lambda sc, task, model: scores.append(dict(sc, task=task, model=model))
    try:
        S.gui_run_task({
            "skill_core": core, "agent": agent, "cfg_fn": lambda: dict(cfg), "background": background,
            "log": log.append, "messagebox": None, "require_arm": lambda: True, "pose_plan": HB.pose_plan(core),
            "slow": HB.slow, "go_q0": lambda label: tw.go_home(label), "start_check": lambda edge: True,
            "safety_gate": lambda: True, "pose_source_mode": lambda: pose_mode, "shared": sh,
            "wrist_selected": lambda: False, "time_fn": lambda: tw.t,
            "sleep_fn": lambda s: tw.step(int(max(0.0, s) / 0.001)),
            "latch_pending": lambda: core.latched, "clear_latch": lambda label: (True, "ok"),
        })
    finally:
        AG._v360_append_score = orig_append
        for k_, v_ in env_keep.items():
            if v_ is None:
                os.environ.pop(k_, None)
            else:
                os.environ[k_] = v_
    state = {iid: SC.item_in_bin(SC.sort_item_centre(body, *tw.object_pose(body))) for iid, body in S.ITEM_IDS.items()}
    return {"log": log, "state": state, "scores": scores, "started": started, "llm": llm, "core": core, "twin": tw}
