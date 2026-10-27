#!/usr/bin/env python3
"""Test double for the V3.13+ ROS monitor helper (same JSON protocol on stdin/stdout), used by test_v3_15*.py.

Robot = config/nero.urdf. TF of watched links = FK of the current joints (optionally with a sign flip on one joint,
FAKE_FLIP_JOINT=3 -> J3 turns the other way, to test the direction wizard). Moves are time-integrated
(speed * FAKE_RAD_S rad/s), cancel stops them. Planning: linear path; path fence check like the real helper;
a goal with any fence point below the top of an applied collision box -> GOAL_IN_COLLISION (-12).
"""
import json, math, os, sys, threading, time
ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
import NERO_V314_Joint_Control as JC  # noqa: E402
import NERO_V315_Consistency_Safety as CS  # noqa: E402

URDF_XML = open(os.path.join(ROOT, "config", "nero.urdf")).read()
URDF = JC.parse_urdf(URDF_XML)
SRDF = ('<robot name="n"><group name="arm"><chain base_link="base_link" tip_link="link7"/></group></robot>')
MODEL = JC.build_model(URDF_XML, SRDF, None, source="fake")
J = MODEL["joints"]
FLIP = int(os.environ.get("FAKE_FLIP_JOINT", "0"))
RAD_S = float(os.environ.get("FAKE_RAD_S", "3.0"))
pos = [float(x) for x in os.environ.get("FAKE_Q0", "0.2,-0.4,0,1.2,0,0.3,0").split(",")]
lock = threading.Lock()
st = {"watch": ["link7"], "scene": [], "move": None, "cancel": False}
stop = threading.Event()
out_lock = threading.Lock()


def emit(**kw):
    kw.setdefault("t", round(time.time(), 3))
    with out_lock:
        print(json.dumps(kw), flush=True)


def tf_q():
    q = list(pos)
    if FLIP:
        q[FLIP - 1] = -q[FLIP - 1]
    return q


def tf_links():
    out = {}
    q = dict(zip(J, tf_q()))
    for ln in st["watch"]:
        T = CS.fk(URDF, q, "base_link", ln)
        out[ln] = CS.pose_of(T) if T else {"error": "no such link"}
        if T:
            out[ln]["xyz"] = [round(v, 6) for v in out[ln]["xyz"]]
    return out


def run_move(mv):
    mid = mv["id"]
    st["cancel"] = False
    emit(event="move_accepted", id=mid)
    with lock:
        q0 = list(pos)
    tgt = [float(x) for x in mv["positions"]]
    names = list(mv["joints"])
    traj = []
    for k in range(21):
        a = k / 20.0
        traj.append([q0[J.index(n)] + a * (t - q0[J.index(n)]) for n, t in zip(names, tgt)])
    two_phase = bool(mv.get("plan_only") or mv.get("fence"))
    info = {"planning_time": 0.05, "trajectory_s": 1.0, "n_points": len(traj)}
    solid = [o for o in st["scene"] if "wall_" not in o["id"]]          # V3.23 walls: checked by the fence
    top = max([o["pos"][2] + o["size"][2] / 2 for o in solid] or [-1e9])
    if solid:
        pts = CS.fence_points(URDF, J, [tgt[names.index(n)] if n in names else q0[i] for i, n in enumerate(J)],
                              "base_link", {"points": ["link4", "link5", "link6", "link7"]}, None)
        if any(p[2] < top for p in pts.values()):
            emit(event="move_result", id=mid, ok=False, error_code=-12, error=JC.moveit_error_text(-12), **info)
            return
    if two_phase and mv.get("fence"):
        viol = CS.path_fence_check(URDF, mv.get("arm_joints") or J, names, traj, mv.get("base") or "base_link",
                                   mv["fence"], mv.get("tool_T"))
        info["fence_checked"] = True
        if viol:
            emit(event="move_result", id=mid, ok=False, error_code=0, fence_violation=viol,
                 error="FENCE: path waypoint %d/%d -> not executed: %s" % (viol["waypoint"] + 1, viol["of"],
                                                                            CS.violation_text(viol["violations"])), **info)
            return
    if mv.get("plan_only"):
        emit(event="move_result", id=mid, ok=True, error_code=1, plan_only=True, **info)
        return
    if two_phase:
        emit(event="move_feedback", id=mid, phase="执行轨迹")
    t0 = time.time()
    rate = RAD_S * float(mv.get("vel", 0.1))
    while True:
        if st["cancel"]:
            time.sleep(0.15)               # deceleration
            emit(event="move_result", id=mid, ok=False, error_code=-7, error=JC.moveit_error_text(-7),
                 elapsed_s=round(time.time() - t0, 2), **info)
            return
        done = True
        with lock:
            for n, t in zip(names, tgt):
                i = J.index(n)
                d = t - pos[i]
                step = rate * 0.02
                if abs(d) > step:
                    pos[i] += step if d > 0 else -step
                    done = False
                else:
                    pos[i] = t
        if done:
            break
        time.sleep(0.02)
    emit(event="move_result", id=mid, ok=True, error_code=1, elapsed_s=round(time.time() - t0, 2), **info)


def reader():
    for line in sys.stdin:
        c = json.loads(line)
        if c.get("quit"):
            stop.set()
            return
        if c.get("move"):
            if st["move"] is not None and st["move"].is_alive():
                emit(event="move_result", id=c["move"]["id"], ok=False, error="another motion is in progress")
                continue
            st["last_move"] = c["move"]
            st["move"] = threading.Thread(target=run_move, args=(c["move"],), daemon=True)
            st["move"].start()
        elif c.get("cancel"):
            st["cancel"] = True
            emit(event="cancel_result", ok=bool(st["move"] and st["move"].is_alive()))
        elif c.get("scene") is not None:
            sc = c["scene"]
            removed = [o["id"] for o in st["scene"] if o["id"] not in [x["id"] for x in sc["objects"]]]
            st["scene"] = list(sc["objects"])
            emit(event="scene_result", id=sc.get("id"), ok=True, added=[o["id"] for o in sc["objects"]], removed=removed,
                 listed_existing=True)
        elif c.get("tf_watch") is not None:
            st["watch"] = list(c["tf_watch"])
        elif c.get("call"):
            emit(event="call_result", id=c.get("id"), call=c["call"], ok=False, error="fake")
        elif c.get("hold"):                     # V3.22: ROS controller sync (recorded for the tests)
            h = c["hold"]
            if os.environ.get("FAKE_HOLD_LOG"):
                with open(os.environ["FAKE_HOLD_LOG"], "a") as f:
                    f.write(json.dumps({"positions": h["positions"], "t": time.time()}) + "\n")
            emit(event="hold_result", id=h.get("id"), ok=True, topic="/arm_controller/joint_trajectory", found=True)


threading.Thread(target=reader, daemon=True).start()
emit(event="monitor_ready")
emit(event="model", model=MODEL)
emit(event="urdf", xml=URDF_XML)
n = 0
while not stop.is_set():
    with lock:
        js = {"names": list(J), "position": [round(v, 6) for v in pos]}
    lf = os.environ.get("FAKE_LJS_FILE")      # V3.22: driver publishes /feedback/leader_joint_states (hand guiding)
    if lf and os.path.exists(lf):
        try:
            lq = [float(v) for v in json.load(open(lf))]
            with lock:
                pos[:] = lq
            emit(event="ljs", joints={"names": list(J), "position": [round(v, 6) for v in lq]})
        except Exception:
            pass
    if not (os.environ.get("FAKE_JS_PAUSE_FILE") and os.path.exists(os.environ["FAKE_JS_PAUSE_FILE"])):
        emit(event="js", joints=js)            # V3.21 tests: joint_states can be paused (leader mode on V111)
    n += 1
    if n % 4 == 0 and not (os.environ.get("FAKE_JS_PAUSE_FILE") and os.path.exists(os.environ["FAKE_JS_PAUSE_FILE"])):
        emit(event="status", js_hz=200.0, joints=js, has_move_group=True, has_move_action=True,
             services={"/emergency_stop": "std_srvs/srv/Trigger"}, gripper_status=None, arm_status={"ctrl_mode": 1},
             tf_links=tf_links(), tcp=tf_links().get("link7"), model_known=True, has_urdf=True, has_srdf=True,
             move=None, last_move_keys=sorted((st.get("last_move") or {}).keys()))
    stop.wait(0.05)
emit(event="monitor_stopped")
