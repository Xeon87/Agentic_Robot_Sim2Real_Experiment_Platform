#!/usr/bin/env python3
# NERO V71.3 HYBRID GPU PLANNER CORE — V2-23 cuRobo + MoveIt replan gate
"""NERO V71 hybrid GPU/BioIK MoveIt2 skill core.

No GUI lives here.  The V70 backend imports this module and exposes the
skills inside the same Tkinter window as the basic controls.

Planning: MoveIt MoveGroup (plan_only)
Execution: arm_controller FollowJointTrajectory DIRECTLY
Approach: MoveIt GetCartesianPath with collision checking
"""

import math
import os
import queue
import json
import socket
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    PlanningSceneComponents,
    OrientationConstraint,
    PositionConstraint,
    JointConstraint,
    RobotTrajectory,
    RobotState,
)
from moveit_msgs.srv import GetCartesianPath, GetPositionIK, GetStateValidity, GetPlanningScene
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

_V71_GPU_CLIENT_SOURCE = "external"
_V71_GPU_CLIENT_IMPORT_ERROR = None
try:
    from NERO_GPU_Planner_Client import GPUPlannerClient
except Exception as _gpu_client_exc:
    # V2-22.1: never lose GPU planning merely because the tiny client module
    # is absent from PYTHONPATH.  Keep a dependency-free localhost client
    # embedded in the ROS runtime as a guaranteed fallback.
    _V71_GPU_CLIENT_SOURCE = "embedded_fallback"
    _V71_GPU_CLIENT_IMPORT_ERROR = repr(_gpu_client_exc)

    class GPUPlannerClient:
        def __init__(self, host=None, port=None, timeout_s=None):
            self.host = str(host or os.environ.get("NERO_GPU_PLANNER_HOST", "127.0.0.1"))
            try:
                self.port = int(port if port is not None else os.environ.get("NERO_GPU_PLANNER_PORT", "8765"))
            except Exception:
                self.port = 8765
            try:
                self.timeout_s = float(timeout_s if timeout_s is not None else os.environ.get("NERO_GPU_PLANNER_TIMEOUT_S", "8.0"))
            except Exception:
                self.timeout_s = 8.0

        def _request(self, payload, timeout_s=None):
            raw = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
            timeout = self.timeout_s if timeout_s is None else float(timeout_s)
            with socket.create_connection((self.host, self.port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                sock.sendall(raw)
                buf = bytearray()
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    if b"\n" in chunk:
                        break
                    if len(buf) > 32 * 1024 * 1024:
                        raise RuntimeError("GPU planner response exceeded 32 MiB")
            if not buf:
                raise RuntimeError("GPU planner returned an empty response")
            out = json.loads(bytes(buf).split(b"\n", 1)[0].decode("utf-8"))
            if not isinstance(out, dict):
                raise RuntimeError("GPU planner returned non-object JSON")
            return out

        def ping(self, timeout_s=0.7):
            try:
                return self._request({"op": "status"}, timeout_s=timeout_s)
            except Exception as exc:
                return {"ok": False, "available": False, "error": str(exc)}

        def plan_pose(self, *, engine, joint_names, q_start, position, quat_xyzw,
                      velocity_scale=0.25, accel_scale=0.25, world=None, max_attempts=5, retry_index=0, random_seed=None,
                      timeout_s=None):
            return self._request({
                "op": "plan_pose",
                "engine": str(engine),
                "joint_names": [str(v) for v in joint_names],
                "q_start": [float(v) for v in q_start],
                "position": [float(v) for v in position],
                "quat_xyzw": [float(v) for v in quat_xyzw],
                "velocity_scale": float(velocity_scale),
                "accel_scale": float(accel_scale),
                "world": world,
                "max_attempts": int(max_attempts),
                "retry_index": int(retry_index),
                "random_seed": None if random_seed is None else int(random_seed),
            }, timeout_s=timeout_s)


OBJECTS = [
    "red_cube",
    "blue_cube",
    "red_ball",
    "blue_ball",
]

# Static sorting bins. Coordinates are in base_link/world coordinates.
# Table top is z=0.000 m in V52, aligned with the robot mounting plane.
BINS = {
    "red_bin":  (-0.455, -0.230, 0.040),
    "blue_bin": (-0.455, +0.230, 0.040),
}

# V70 conservative initial-object zone. Future spawned/manually reset objects should
# start inside this rectangle. It is intentionally smaller than the full arm
# workspace: the four V45/V66-proven tabletop grasp locations lie around this
# region, and the V70 initial objects are placed with extra margin from its edges.
# Coordinates are in base_link/world frame on the z=0 tabletop.
OBJECT_ZONE = {
    "x_min": -0.475,
    "x_max": -0.335,
    "y_min": -0.110,
    "y_max": +0.110,
}
OBJECT_ZONE_CENTER = (-0.405, 0.0)
OBJECT_ZONE_SIZE = (0.140, 0.220)
OBJECT_ZONE_BIN_GAP = 0.050  # 50 mm to the nearest physical bin wall in Y

BIN_RELEASE_CLEARANCE = 0.120

MOVEIT_ERROR_NAMES = {
    1: "SUCCESS",
    99999: "FAILURE",
    -1: "PLANNING_FAILED",
    -2: "INVALID_MOTION_PLAN",
    -3: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
    -4: "CONTROL_FAILED",
    -5: "UNABLE_TO_ACQUIRE_SENSOR_DATA",
    -6: "TIMED_OUT",
    -7: "PREEMPTED",
    -10: "START_STATE_IN_COLLISION",
    -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
    -12: "GOAL_IN_COLLISION",
    -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
    -14: "GOAL_CONSTRAINTS_VIOLATED",
    -15: "INVALID_GROUP_NAME",
    -16: "INVALID_GOAL_CONSTRAINTS",
    -17: "INVALID_ROBOT_STATE",
    -18: "INVALID_LINK_NAME",
    -19: "INVALID_OBJECT_NAME",
    -21: "FRAME_TRANSFORM_FAILURE",
    -22: "COLLISION_CHECKING_UNAVAILABLE",
    -23: "ROBOT_STATE_STALE",
    -24: "SENSOR_INFO_STALE",
    -25: "COMMUNICATION_FAILURE",
    -29: "CRASH",
    -30: "ABORT",
    -31: "NO_IK_SOLUTION",
}

FJT_ERROR_NAMES = {
    0: "SUCCESSFUL",
    -1: "INVALID_GOAL",
    -2: "INVALID_JOINTS",
    -3: "OLD_HEADER_TIMESTAMP",
    -4: "PATH_TOLERANCE_VIOLATED",
    -5: "GOAL_TOLERANCE_VIOLATED",
}

GRASP_Z_OFFSET = {
    "red_cube": 0.025,
    "blue_cube": 0.025,
    "red_ball": 0.030,
    "blue_ball": 0.030,
}
DEFAULT_GRASP_Z_OFFSET = 0.028
MOVE_ABOVE_HEIGHT = 0.120
CARTESIAN_MAX_STEP = 0.005
CARTESIAN_SPEED = 0.035


def quat_normalize(q):
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x/n, y/n, z/n, w/n)


def quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    )


def quat_rotate_vector(q, v):
    """Rotate a 3-vector by xyzw quaternion q."""
    qn = quat_normalize(q)
    qv = (float(v[0]), float(v[1]), float(v[2]), 0.0)
    qc = (-qn[0], -qn[1], -qn[2], qn[3])
    r = quat_multiply(quat_multiply(qn, qv), qc)
    return (float(r[0]), float(r[1]), float(r[2]))


def quat_axis_angle(axis, angle):
    x, y, z = axis
    n = math.sqrt(x*x + y*y + z*z)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    x, y, z = x/n, y/n, z/n
    h = 0.5 * float(angle)
    sh = math.sin(h)
    return (x*sh, y*sh, z*sh, math.cos(h))


def top_down_quat(yaw_rad=0.0, tilt_axis=None, tilt_rad=0.0):
    """tcp +Z points downward; optional small world-frame tilt."""
    q_down = (1.0, 0.0, 0.0, 0.0)  # Rx(pi)
    q_yaw = quat_axis_angle((0.0, 0.0, 1.0), yaw_rad)
    q = quat_multiply(q_yaw, q_down)

    if tilt_axis is not None and abs(float(tilt_rad)) > 1e-9:
        q_tilt = quat_axis_angle(tilt_axis, tilt_rad)
        q = quat_multiply(q_tilt, q)

    return quat_normalize(q)


def quat_with_world_tilt(base_q, roll_deg=0.0, pitch_deg=0.0):
    """Apply a bounded world-frame X/Y tilt to a reference TCP quaternion.

    V70 uses this only as a *local* orientation relaxation around the proven
    grasp orientation.  It never changes yaw and never permits a large wrist
    flip.
    """
    qr = quat_axis_angle((1.0, 0.0, 0.0), math.radians(float(roll_deg)))
    qp = quat_axis_angle((0.0, 1.0, 0.0), math.radians(float(pitch_deg)))
    qtilt = quat_multiply(qp, qr)
    return quat_normalize(quat_multiply(qtilt, quat_normalize(base_q)))


ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]

# V70 diagnostic copy of the exact NERO/V31 hard joint limits used by the
# MuJoCo backend.  These are used only to classify/log IK frontiers; they do
# not replace URDF/MoveIt enforcement.
JOINT_LIMITS_RAD = [
    (-2.70526,  2.70526),
    (-1.74,     1.74),
    (-2.75,     2.75),
    (-1.01,     2.14),
    (-2.75,     2.75),
    (-0.73,     0.95),
    (-1.5708,   1.5708),
]
JOINT_LIMIT_LIKELY_MARGIN_DEG = 5.0
IK_TIMEOUT_S = 0.25
IK_HEIGHT_CANDIDATES = (0.120, 0.180, 0.240)
IK_YAW_CANDIDATES = (
    0.0,
    math.pi / 2.0,
    -math.pi / 2.0,
    math.pi,
)
IK_SMALL_TILT_DEG = 15.0

# V70 natural-motion policy.  Hard collision/joint-limit enforcement remains in
# MoveIt/URDF; these constants only rank already collision-aware candidates.
V70_POSE_IK_TOP_K = 3
V70_POSE_IK_MAX_SEEDS = 10
V70_DEDUPE_JOINT_DEG = 0.50
V70_REORIENT_EXTRA_SEED_DEG = (8.0, 15.0, 24.0)

# V71 hybrid planner selection.  GPU planning is optional and out-of-process;
# absence/failure never prevents the proven BioIK/TRAC stack from starting.
V71_PLANNER_MODES = ("GPU Auto", "cuMotion", "cuMotion Strict", "cuRobo", "cuRobo Strict", "BioIK/MoveIt")
V71_DEFAULT_PLANNER_MODE = os.environ.get("NERO_PLANNER_MODE", "cuRobo Strict")
V71_GPU_MAX_POINTS = 600
V71_GPU_REPLAN_ATTEMPTS = max(10, int(os.environ.get("NERO_V23_CUROBO_REPLAN_ATTEMPTS", "10")))


# ============================================================
# V52 TRAC-IK DIAGNOSTIC LAYER
# The actual IK algorithm is loaded inside MoveIt from kinematics.yaml.
# These methods only exercise /compute_ik and report what the active
# MoveIt solver is doing.  Legacy V48 skill semantics stay intact.
# ============================================================

TRAC_PANEL_DEFAULT_TIMEOUT_S = 0.10
TRAC_BENT_SEED = (
    0.0,
    0.40,
    0.0,
    0.20,
    0.0,
    0.0,
    0.0,
)


def moveit_error_text(error_code):
    code = int(getattr(error_code, "val", error_code))
    name = MOVEIT_ERROR_NAMES.get(code, f"UNKNOWN_{code}")
    msg = getattr(error_code, "message", "")
    src = getattr(error_code, "source", "")
    extra = []
    if msg:
        extra.append(str(msg))
    if src:
        extra.append(f"source={src}")
    return f"{name} ({code})" + ((" | " + " | ".join(extra)) if extra else "")


class NeroMoveItCore:
    def __init__(
        self,
        set_arm_command_gate=None,
        get_safety_trip_count=None,
        request_soft_recover=None,
    ):
        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = rclpy.create_node("nero_v66_dynamics_safe_bioik_moveit_core")

        self.move_client = ActionClient(self.node, MoveGroup, "/move_action")
        self.arm_controller_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )
        self.cartesian_client = self.node.create_client(
            GetCartesianPath,
            "/compute_cartesian_path",
        )

        # Primary IK service: root MoveIt instance configured with BioIK.
        self.ik_client = self.node.create_client(
            GetPositionIK,
            "/compute_ik",
        )

        # Secondary reference/fallback IK service: a second MoveIt instance
        # launched in /trac_ref and configured with TRAC-IK.  It is used for
        # A/B diagnostics and as a bounded fallback inside the V63 incremental
        # planner; physical execution still goes through the single root
        # arm_controller.
        self.trac_ik_client = self.node.create_client(
            GetPositionIK,
            "/trac_ref/compute_ik",
        )

        # V71 GPU trajectories are independently re-validated by MoveIt's current
        # planning scene before execution.  This is intentionally in the ROS
        # process rather than trusting an external GPU daemon.
        self.state_validity_client = self.node.create_client(
            GetStateValidity,
            "/check_state_validity",
        )
        self.planning_scene_client = self.node.create_client(
            GetPlanningScene,
            "/get_planning_scene",
        )
        self._v71_world_cache_time = 0.0
        self._v71_world_cache = None
        self._planner_mode = (
            V71_DEFAULT_PLANNER_MODE
            if V71_DEFAULT_PLANNER_MODE in V71_PLANNER_MODES
            else "GPU Auto"
        )
        self._gpu_client = GPUPlannerClient()
        self._last_gpu_status = {"ok": False, "available": False, "error": "not probed"}

        self._joint_state_lock = threading.Lock()
        self._latest_joint_state = None

        self.feedback_sub = self.node.create_subscription(
            JointState,
            "/feedback/joint_states",
            self._feedback_joint_state_cb,
            20,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        self._log_queue = queue.Queue()
        self._state_lock = threading.Lock()
        self._busy = False
        self._active_goal_handle = None
        self._spin_thread = None
        self._shutdown = False
        self.active_grasp_quat = None

        # V53 recovery generation. A hard reset/recover invalidates the
        # currently running worker immediately so an old failed skill cannot
        # keep the console BUSY or resume commanding after recovery.
        self._recovery_epoch = 0
        self._active_worker_epoch = None
        self._worker_local = threading.local()
        self._set_arm_command_gate_cb = set_arm_command_gate
        self._get_safety_trip_count_cb = get_safety_trip_count
        self._request_soft_recover_cb = request_soft_recover

        self.emit("[OK] ROS2/rclpy skill core node created: /nero_v66_dynamics_safe_bioik_moveit_core")
        self.emit("[WAIT] Waiting for MoveIt actions/services...")
        self.emit(
            f"[V71 PLANNER] default={self._planner_mode} | GPU daemon optional | "
            "BioIK/MoveIt fallback preserved; TRAC-IK preserved"
        )
        self.emit(
            f"[V2-23 GPU CLIENT] source={_V71_GPU_CLIENT_SOURCE}"
            + (f" | external_import_error={_V71_GPU_CLIENT_IMPORT_ERROR}" if _V71_GPU_CLIENT_IMPORT_ERROR else "")
        )

    def _feedback_joint_state_cb(self, msg):
        with self._joint_state_lock:
            copy = JointState()
            copy.header = msg.header
            copy.name = list(msg.name)
            copy.position = list(msg.position)
            copy.velocity = list(msg.velocity)
            copy.effort = list(msg.effort)
            self._latest_joint_state = copy

    def _current_joint_state_copy(self, timeout_s=2.0):
        deadline = time.time() + float(timeout_s)

        while time.time() < deadline:
            with self._joint_state_lock:
                js = self._latest_joint_state

                if js is not None:
                    out = JointState()
                    out.header = js.header
                    out.name = list(js.name)
                    out.position = list(js.position)
                    out.velocity = list(js.velocity)
                    out.effort = list(js.effort)
                    return out

            time.sleep(0.02)

        raise RuntimeError(
            "No /feedback/joint_states received for IK seed"
        )

    def start(self):
        if self._spin_thread is not None:
            return
        self._spin_thread = threading.Thread(
            target=rclpy.spin,
            args=(self.node,),
            daemon=True,
            name="v63_moveit_spin",
        )
        self._spin_thread.start()
        self.emit("[OK] ROS2 executor thread started")

        if self.ik_client.wait_for_service(timeout_sec=0.2):
            self.emit("[OK] BioIK service READY: /compute_ik")
        else:
            self.emit("[WAIT] BioIK service not ready yet: /compute_ik")

        if self.trac_ik_client.wait_for_service(timeout_sec=0.2):
            self.emit("[OK] TRAC reference service READY: /trac_ref/compute_ik")
        else:
            self.emit("[WAIT] TRAC reference service not ready yet: /trac_ref/compute_ik")

    def shutdown(self):
        self._shutdown = True
        try:
            self.cancel_active_motion()
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    def emit(self, text):
        stamp = time.strftime("%H:%M:%S")
        self._log_queue.put(f"[{stamp}] {text}")

    def drain_logs(self, limit=100):
        out = []
        for _ in range(limit):
            try:
                out.append(self._log_queue.get_nowait())
            except queue.Empty:
                break
        return out

    @property
    def busy(self):
        with self._state_lock:
            return self._busy

    def _set_busy(self, value):
        with self._state_lock:
            self._busy = bool(value)

    def submit(self, fn, *args):
        with self._state_lock:
            if self._busy:
                self.emit("BUSY — wait for current skill or press STOP + RECOVER")
                return False
            worker_epoch = int(self._recovery_epoch)
            self._busy = True
            self._active_worker_epoch = worker_epoch

        def worker():
            self._worker_local.epoch = worker_epoch
            try:
                fn(*args)
            except Exception as exc:
                # Recovery intentionally aborts the stale worker; make that
                # visible but do not let it poison the next skill.
                self.emit(f"FAILED: {exc}")
            finally:
                with self._state_lock:
                    if self._active_worker_epoch == worker_epoch:
                        self._active_goal_handle = None
                        self._active_worker_epoch = None
                        self._busy = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    def readiness(self):
        try:
            move = bool(self.move_client.server_is_ready())
            arm = bool(self.arm_controller_client.server_is_ready())
            cart = bool(self.cartesian_client.service_is_ready())
            return move, arm, cart
        except Exception:
            return False, False, False

    def wait_future(self, future, timeout_s, label):
        deadline = time.time() + float(timeout_s)
        worker_epoch = getattr(self._worker_local, "epoch", None)
        while time.time() < deadline:
            if worker_epoch is not None:
                with self._state_lock:
                    current_epoch = int(self._recovery_epoch)
                if int(worker_epoch) != current_epoch:
                    raise RuntimeError("skill aborted by recovery/reset")
            if future.done():
                return future.result()
            if self._shutdown:
                raise RuntimeError("Skill core shutting down")
            time.sleep(0.01)
        raise RuntimeError(f"{label} timeout after {timeout_s:.1f}s")

    def cancel_active_motion(self):
        with self._state_lock:
            gh = self._active_goal_handle
        if gh is None:
            return
        try:
            gh.cancel_goal_async()
            self.emit("Cancel requested for active arm trajectory")
        except Exception as exc:
            self.emit(f"Cancel request failed: {exc}")

    def _set_arm_command_gate(self, enabled):
        if self._set_arm_command_gate_cb is None:
            return
        try:
            self._set_arm_command_gate_cb(bool(enabled))
        except Exception as exc:
            self.emit(f"[WARN] arm command gate callback failed: {exc}")

    def _safety_trip_count(self):
        if self._get_safety_trip_count_cb is None:
            return 0
        try:
            return int(self._get_safety_trip_count_cb())
        except Exception:
            return 0

    def _request_soft_recover(self):
        if self._request_soft_recover_cb is None:
            return
        try:
            self._request_soft_recover_cb()
        except Exception as exc:
            self.emit(f"[WARN] soft recovery request failed: {exc}")

    def _verify_physical_joint_target(
        self, robot_trajectory, timeout_s=2.5, tolerance_rad=0.10
    ):
        """Verify that MuJoCo actually reached the controller's final J1..J7 target.

        The ros2_control FakeSystem can report FollowJointTrajectory SUCCESS even
        when the physical MuJoCo bridge was stopped by the secondary safety guard.
        V59 never reports execution success until real /feedback/joint_states agrees.
        """
        jt = robot_trajectory.joint_trajectory
        if not jt.points or not jt.joint_names:
            return True

        final = list(jt.points[-1].positions)
        names = list(jt.joint_names)
        if len(final) < len(names):
            return True

        target = {str(n): float(p) for n, p in zip(names, final)}
        deadline = time.time() + float(timeout_s)
        last_max = float("inf")
        last_rms = float("inf")

        while time.time() < deadline:
            js = self._current_joint_state_copy(timeout_s=0.25)
            actual = {str(n): float(p) for n, p in zip(js.name, js.position)}
            errs = [
                abs(actual[n] - target[n])
                for n in names
                if n in actual and n in target
            ]
            if len(errs) == len(names):
                last_max = max(errs) if errs else 0.0
                last_rms = math.sqrt(sum(e * e for e in errs) / max(1, len(errs)))
                if last_max <= float(tolerance_rad):
                    self.emit(
                        f"[EXEC VERIFY OK] physical max joint error "
                        f"{math.degrees(last_max):.2f}deg | "
                        f"rms {math.degrees(last_rms):.2f}deg"
                    )
                    return True
            time.sleep(0.04)

        self.emit(
            f"[EXEC VERIFY FAILED] physical max joint error "
            f"{math.degrees(last_max):.1f}deg | "
            f"rms {math.degrees(last_rms):.1f}deg"
        )
        raise RuntimeError(
            "PHYSICAL TRACKING MISMATCH: ros2_control reported success but "
            f"MuJoCo is still {math.degrees(last_max):.1f}deg from the planned joint goal"
        )

    def force_recover(self, reason="manual recovery"):
        # Invalidate the running worker first. This prevents stale async
        # results from re-entering execution after a reset.
        with self._state_lock:
            gh = self._active_goal_handle
            self._recovery_epoch += 1
            self._active_goal_handle = None
            self._active_worker_epoch = None
            self._busy = False

        if gh is not None:
            try:
                gh.cancel_goal_async()
            except Exception as exc:
                self.emit(f"Cancel request failed during recovery: {exc}")

        self._set_arm_command_gate(False)
        self.active_grasp_quat = None
        self.emit(
            f"[RECOVER] {reason}; stale worker invalidated; "
            "ROS arm gate OFF; grasp branch cleared"
        )
        return True

    def clear_skill_state(self):
        self.force_recover("skill state cleared")

    # ============================================================
    # V3.05 EXECUTION-LAYER HEALTH
    # ------------------------------------------------------------
    # V3.04 log: every trajectory "succeeded" within ~0.4 s while MuJoCo was
    # still 105-136 deg from the goal, single-point sync goals were rejected,
    # and the arm later drifted to an OLD trajectory's end pose.  That pattern
    # matches a goal being cancelled/pre-empted (FollowJointTrajectory then
    # still carries error_code=0) and/or two /arm_controller instances.
    # V3.05: (a) refuse to execute when more than one action server answers
    # for /arm_controller; (b) accept a result only if GoalStatus==SUCCEEDED.
    # ============================================================
    V305_GOAL_STATUS_NAMES = {0: "UNKNOWN", 1: "ACCEPTED", 2: "EXECUTING", 3: "CANCELING",
                              4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}

    def _v305_check_single_arm_controller(self):
        if str(os.environ.get("NERO_V305_SINGLE_CONTROLLER_CHECK", "1")) == "0":
            return
        topics = {
            "action_status": "/arm_controller/follow_joint_trajectory/_action/status",
            "controller_state": "/arm_controller/controller_state",
        }
        counts = {}
        for key, topic in topics.items():
            try:
                counts[key] = int(self.node.count_publishers(topic))
            except Exception:
                counts[key] = -1
        if counts.get("action_status", 0) > 1 or counts.get("controller_state", 0) > 1:
            self.emit(
                f"[V3.05 DUPLICATE ARM CONTROLLER] publishers={counts} -> refusing to execute. "
                "Stale ros2_control/move_group processes from an earlier run are probably still alive; "
                "close everything (pkill -f ros2_control_node; pkill -f move_group) and restart."
            )
            raise RuntimeError(
                f"EXECUTION_LAYER_FAULT: DUPLICATE_ARM_CONTROLLER publishers={counts}"
            )

    def _v305_require_succeeded(self, wrapped, label):
        status = int(getattr(wrapped, "status", 4))
        if status != 4:
            name = self.V305_GOAL_STATUS_NAMES.get(status, f"STATUS_{status}")
            msg = ""
            try:
                msg = str(getattr(wrapped.result, "error_string", "") or "")
            except Exception:
                pass
            self.emit(f"[V3.05 GOAL NOT SUCCEEDED] {label} | status={name} | {msg}")
            raise RuntimeError(
                f"EXECUTION_LAYER_FAULT: {label} goal ended with status={name} (not SUCCEEDED) {msg}".strip()
            )

    def _sync_arm_controller_to_feedback(self):
        """
        Synchronize ros2_control's mock arm state to the actual MuJoCo
        feedback while the backend command gate is OFF.  This prevents a
        later trajectory from starting from an old controller-held pose.
        """
        if not self.arm_controller_client.wait_for_server(timeout_sec=3.0):
            self.emit(
                "[EXECUTION INTERFACE LOST] /arm_controller/follow_joint_trajectory "
                "is not available. Planning is healthy, but execution is blocked. "
                "Do not retry motion until the root arm_controller is ACTIVE."
            )
            raise RuntimeError(
                "/arm_controller/follow_joint_trajectory unavailable"
            )

        js = self._current_joint_state_copy()
        name_to_pos = {
            str(n): float(p)
            for n, p in zip(js.name, js.position)
        }

        arm_names = [f"joint{i}" for i in range(1, 8)]
        if not all(name in name_to_pos for name in arm_names):
            raise RuntimeError(
                "Actual feedback missing one or more arm joints"
            )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = arm_names

        pt = JointTrajectoryPoint()
        pt.positions = [name_to_pos[name] for name in arm_names]
        pt.velocities = [0.0] * 7
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 250_000_000
        goal.trajectory.points = [pt]

        self.emit(
            "Synchronizing ros2_control arm state to actual MuJoCo pose..."
        )

        gh = self.wait_future(
            self.arm_controller_client.send_goal_async(goal),
            5.0,
            "arm controller sync goal response",
        )

        if gh is None or not gh.accepted:
            raise RuntimeError("EXECUTION_LAYER_FAULT: arm controller rejected sync trajectory")

        wrapped = self.wait_future(
            gh.get_result_async(),
            8.0,
            "arm controller sync",
        )

        if wrapped is None:
            raise RuntimeError("arm controller sync returned no result")

        code = int(wrapped.result.error_code)
        if code != 0:
            name = FJT_ERROR_NAMES.get(code, f"UNKNOWN_{code}")
            raise RuntimeError(
                f"arm controller sync failed: {name} ({code})"
            )
        self._v305_require_succeeded(wrapped, "sync")

        self.emit("ros2_control arm state synchronized")

    def wait_for_tf(self, child, timeout_s=2.0):
        deadline = time.time() + float(timeout_s)
        last_exc = None
        while time.time() < deadline:
            try:
                return self.tf_buffer.lookup_transform(
                    "base_link",
                    child,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.2),
                )
            except Exception as exc:
                last_exc = exc
                time.sleep(0.05)
        raise RuntimeError(f"TF unavailable: base_link -> {child}: {last_exc}")

    def current_tcp_pose(self):
        tfm = self.wait_for_tf("tcp_link")
        p = tfm.transform.translation
        q = tfm.transform.rotation
        return [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]

    def object_pose(self, object_name):
        tfm = self.wait_for_tf(f"sim_obj_{object_name}")
        p = tfm.transform.translation
        q = tfm.transform.rotation
        return [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]

    def observe_scene(self):
        tcp_xyz, _ = self.current_tcp_pose()
        lines = [
            f"tcp_link: [{tcp_xyz[0]:+.3f}, {tcp_xyz[1]:+.3f}, {tcp_xyz[2]:+.3f}]"
        ]
        for name in OBJECTS:
            try:
                xyz, _ = self.object_pose(name)
                lines.append(
                    f"{name:18s} [{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}]"
                )
            except Exception:
                lines.append(f"{name:18s} TF unavailable")
        lines.append(
            "object_zone        "
            f"x=[{OBJECT_ZONE['x_min']:+.3f},{OBJECT_ZONE['x_max']:+.3f}] "
            f"y=[{OBJECT_ZONE['y_min']:+.3f},{OBJECT_ZONE['y_max']:+.3f}] "
            f"size={OBJECT_ZONE_SIZE[0]*1000:.0f}x{OBJECT_ZONE_SIZE[1]*1000:.0f}mm"
        )
        lines.append(
            f"red_bin center     [{BINS['red_bin'][0]:+.3f}, {BINS['red_bin'][1]:+.3f}]"
        )
        lines.append(
            f"blue_bin center    [{BINS['blue_bin'][0]:+.3f}, {BINS['blue_bin'][1]:+.3f}]"
        )
        self.emit("\n".join(lines))
        return lines

    def _pose_goal_constraints(
        self,
        xyz,
        quat_xyzw=None,
        position_tolerance=0.010,
        orientation_tolerance=0.20,
    ):
        constraints = Constraints()

        pc = PositionConstraint()
        pc.header.frame_id = "base_link"
        pc.link_name = "tcp_link"
        pc.weight = 1.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [float(position_tolerance)]

        region_pose = PoseStamped()
        region_pose.header.frame_id = "base_link"
        region_pose.pose.position.x = float(xyz[0])
        region_pose.pose.position.y = float(xyz[1])
        region_pose.pose.position.z = float(xyz[2])
        region_pose.pose.orientation.w = 1.0

        pc.constraint_region.primitives = [sphere]
        pc.constraint_region.primitive_poses = [region_pose.pose]
        constraints.position_constraints = [pc]

        if quat_xyzw is not None:
            q = quat_normalize(quat_xyzw)
            oc = OrientationConstraint()
            oc.header.frame_id = "base_link"
            oc.link_name = "tcp_link"
            oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = q
            oc.absolute_x_axis_tolerance = float(orientation_tolerance)
            oc.absolute_y_axis_tolerance = float(orientation_tolerance)
            oc.absolute_z_axis_tolerance = float(orientation_tolerance)
            oc.weight = 1.0
            constraints.orientation_constraints = [oc]

        return constraints

    def _build_move_group_goal(
        self,
        xyz,
        quat_xyzw,
        velocity_scale,
        accel_scale,
        orientation_tolerance,
        position_tolerance,
        planner_id="",
        planning_attempts=8,
        planning_time=8.0,
    ):
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = "arm"
        req.pipeline_id = "ompl"
        if planner_id:
            req.planner_id = str(planner_id)
        req.num_planning_attempts = int(planning_attempts)
        req.allowed_planning_time = float(planning_time)
        req.max_velocity_scaling_factor = float(velocity_scale)
        req.max_acceleration_scaling_factor = float(accel_scale)
        req.start_state.is_diff = True

        req.workspace_parameters.header.frame_id = "base_link"
        req.workspace_parameters.min_corner.x = -1.5
        req.workspace_parameters.min_corner.y = -1.5
        req.workspace_parameters.min_corner.z = -0.2
        req.workspace_parameters.max_corner.x = 1.5
        req.workspace_parameters.max_corner.y = 1.5
        req.workspace_parameters.max_corner.z = 1.8

        req.goal_constraints = [
            self._pose_goal_constraints(
                xyz,
                quat_xyzw,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
            )
        ]

        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        return goal

    # ========================================================
    # V70 NATURAL-MOTION / JOINT-RESERVE-AWARE IK RANKING
    # ========================================================
    def _v70_joint_motion_metrics(self, current_arm, candidate_arm):
        """Return normalized motion + reserve metrics for one IK candidate.

        The score is intentionally kinematic and deterministic.  It does not
        weaken collision checking and it does not replace MoveIt.  It only
        chooses among collision-aware IK solutions before OMPL planning.
        """
        current=[float(v) for v in current_arm]
        cand=[float(v) for v in candidate_arm]
        delta=[abs(cand[i]-current[i]) for i in range(7)]
        ranges=[max(1e-9,float(hi)-float(lo)) for lo,hi in JOINT_LIMITS_RAD]
        delta_norm=[delta[i]/ranges[i] for i in range(7)]

        reserve_norm=[]
        reserve_deg=[]
        for i,(lo,hi) in enumerate(JOINT_LIMITS_RAD):
            margin=max(0.0,min(cand[i]-float(lo),float(hi)-cand[i]))
            half=max(1e-9,0.5*(float(hi)-float(lo)))
            reserve_norm.append(margin/half)
            reserve_deg.append(math.degrees(margin))

        wrist_idx=(4,5,6)
        return {
            "delta_rad":delta,
            "delta_deg":[math.degrees(v) for v in delta],
            "delta_norm":delta_norm,
            "max_delta_norm":max(delta_norm),
            "max_wrist_delta_norm":max(delta_norm[i] for i in wrist_idx),
            "sum_delta_norm":sum(delta_norm),
            "sum_delta_deg":sum(math.degrees(v) for v in delta),
            "reserve_norm":reserve_norm,
            "reserve_deg":reserve_deg,
            "min_reserve_norm":min(reserve_norm),
            "min_wrist_reserve_norm":min(reserve_norm[i] for i in wrist_idx),
            "min_reserve_deg":min(reserve_deg),
            "min_wrist_reserve_deg":min(reserve_deg[i] for i in wrist_idx),
        }

    def _v70_endpoint_score(self, current_arm, candidate_arm):
        """Lexicographic endpoint score: minimax travel -> wrist reserve -> total travel."""
        m=self._v70_joint_motion_metrics(current_arm,candidate_arm)
        # First avoid one joint doing something visually/kinematically extreme.
        # Then preserve J5-J7 reserve, then minimize total normalized travel.
        score=(
            float(m["max_delta_norm"]),
            float(m["max_wrist_delta_norm"]),
            -float(m["min_wrist_reserve_norm"]),
            float(m["sum_delta_norm"]),
            -float(m["min_reserve_norm"]),
        )
        return score,m

    @staticmethod
    def _v70_clamp_seed_arm(arm):
        out=[]
        guard=math.radians(0.20)
        for q,(lo,hi) in zip(arm,JOINT_LIMITS_RAD):
            out.append(max(float(lo)+guard,min(float(hi)-guard,float(q))))
        return out

    def _v70_natural_seed_variants(self, base_seed, base_arm, reorientation=False):
        """Generate deterministic seeds on both sides of the redundant wrist branch.

        BioIK is still collision-aware.  Seeds merely ask the redundant 7-DOF
        solver to expose more than one branch so V70 can rank them instead of
        accepting the first result it happens to return.
        """
        base=[float(v) for v in base_arm]
        centers=[0.5*(float(lo)+float(hi)) for lo,hi in JOINT_LIMITS_RAD]
        arms=[]
        tags=[]

        def add(tag,arm):
            arm=self._v70_clamp_seed_arm(arm)
            key=tuple(round(float(v),7) for v in arm)
            if key in seen:
                return
            seen.add(key); tags.append(tag); arms.append(arm)

        seen=set()
        add('current',base)

        # Pull only the wrist toward the center by several amounts.  This is the
        # direct antidote to an unnecessary half-turn consuming later wrist reserve.
        for frac in (0.25,0.50,0.75):
            arm=list(base)
            for i in (4,5,6):
                arm[i]=base[i]+frac*(centers[i]-base[i])
            add(f'wrist_center_{int(frac*100)}',arm)

        # A small whole-arm center bias gives BioIK an elbow/shoulder alternative.
        for frac in (0.15,0.30):
            arm=[base[i]+frac*(centers[i]-base[i]) for i in range(7)]
            add(f'all_center_{int(frac*100)}',arm)

        # Opposite-sign wrist seeds explicitly expose clockwise/counter-clockwise
        # redundant branches.  The final solution is still required to satisfy the
        # exact Cartesian pose and MoveIt collision checking.
        degrees=(8.0,15.0) if not reorientation else V70_REORIENT_EXTRA_SEED_DEG
        for d in degrees:
            r=math.radians(float(d))
            for sign in (+1.0,-1.0):
                arm=list(base)
                arm[5]+=sign*r
                arm[6]-=sign*r
                add(f'J6{sign*d:+.0f}_J7{-sign*d:+.0f}',arm)
                arm=list(base)
                arm[4]+=sign*0.65*r
                arm[5]-=sign*r
                arm[6]+=sign*0.65*r
                add(f'wrist_opposite_{sign*d:+.0f}',arm)

        out=[]
        for tag,arm in zip(tags,arms):
            out.append((tag,self._seed_with_arm_positions(base_seed,arm)))
            if len(out)>=int(V70_POSE_IK_MAX_SEEDS) and not reorientation:
                break
        return out

    def _v70_reorientation_seed_variants(self, base_seed, base_arm):
        """Adapter used by V2-21/V2-15 reorientation search.

        Returns the historical (p5,p6,p7,JointState) shape so upper-layer motion
        code can add wider clockwise/counter-clockwise probes without changing
        grasp/placement semantics.
        """
        base=[float(v) for v in base_arm]
        out=[]
        for tag,seed in self._v70_natural_seed_variants(base_seed,base,reorientation=True):
            arm=self._arm_positions_from_joint_state(seed)
            d=[math.degrees(arm[i]-base[i]) for i in range(7)]
            out.append((float(d[4]),float(d[5]),float(d[6]),seed))
        return out

    def _v70_ranked_pose_ik_candidates(self, xyz, quat_xyzw, max_candidates=V70_POSE_IK_TOP_K,
                                       timeout_s=0.14):
        """Collect/rank multiple collision-aware BioIK solutions for one pose."""
        base_seed=self._current_joint_state_copy()
        current=self._arm_positions_from_joint_state(base_seed)
        raw=[]
        errors=[]
        for tag,seed in self._v70_natural_seed_variants(base_seed,current,reorientation=False):
            try:
                joints,err=self._compute_ik(
                    xyz,quat_xyzw,timeout_s=float(timeout_s),
                    avoid_collisions=True,seed_joint_state=seed,
                )
            except Exception as exc:
                errors.append(f'{tag}:EXCEPTION:{exc}')
                continue
            if joints is None:
                errors.append(f'{tag}:{err}')
                continue

            # Deduplicate solver results that differ only by tiny numerical noise.
            duplicate=False
            for _s,old,_tag,_m in raw:
                if max(abs(float(joints[i])-float(old[i])) for i in range(7)) <= math.radians(V70_DEDUPE_JOINT_DEG):
                    duplicate=True; break
            if duplicate:
                continue
            score,metrics=self._v70_endpoint_score(current,joints)
            raw.append((score,[float(v) for v in joints],tag,metrics))

        raw.sort(key=lambda item:item[0])
        if raw:
            for rank,item in enumerate(raw[:max(1,int(max_candidates))],start=1):
                score,joints,tag,m=item
                self.emit(
                    f"[V70 IK CANDIDATE] rank={rank} seed={tag} | "
                    f"max_dJ={max(m['delta_deg']):.1f}deg | total_dJ={m['sum_delta_deg']:.1f}deg | "
                    f"wrist_reserve={m['min_wrist_reserve_deg']:.1f}deg | all_reserve={m['min_reserve_deg']:.1f}deg"
                )
        else:
            self.emit(f"[V70 IK CANDIDATE] none | errors_tail={errors[-3:]}")
        return raw[:max(1,int(max_candidates))],errors,current

    def _v70_trajectory_metrics(self, traj, current_arm, goal_arm):
        """Measure actual planned joint travel, including detours/winding."""
        direct=self._v70_joint_motion_metrics(current_arm,goal_arm)
        if traj is None or not getattr(traj,'joint_trajectory',None):
            return {
                'max_cumulative_norm':direct['max_delta_norm'],
                'total_cumulative_norm':direct['sum_delta_norm'],
                'max_extra_norm':0.0,
                'cumulative_deg':list(direct['delta_deg']),
                'direct':direct,
            }
        jt=traj.joint_trajectory
        names=list(jt.joint_names)
        idx={str(n):i for i,n in enumerate(names)}
        if not all(n in idx for n in ARM_JOINT_NAMES):
            return {
                'max_cumulative_norm':999.0,'total_cumulative_norm':999.0,
                'max_extra_norm':999.0,'cumulative_deg':[999.0]*7,'direct':direct,
            }
        prev=[float(v) for v in current_arm]
        cumulative=[0.0]*7
        for pt in jt.points:
            q=[float(pt.positions[idx[n]]) for n in ARM_JOINT_NAMES]
            for i in range(7): cumulative[i]+=abs(q[i]-prev[i])
            prev=q
        ranges=[max(1e-9,float(hi)-float(lo)) for lo,hi in JOINT_LIMITS_RAD]
        cum_norm=[cumulative[i]/ranges[i] for i in range(7)]
        extra_norm=[max(0.0,cum_norm[i]-direct['delta_norm'][i]) for i in range(7)]
        return {
            'max_cumulative_norm':max(cum_norm),
            'total_cumulative_norm':sum(cum_norm),
            'max_extra_norm':max(extra_norm),
            'cumulative_deg':[math.degrees(v) for v in cumulative],
            'direct':direct,
        }

    # ============================================================
    # V71 HYBRID GPU PLANNER DISPATCH
    # ============================================================
    def planner_modes(self):
        return list(V71_PLANNER_MODES)

    def get_planner_mode(self):
        return str(self._planner_mode)

    def set_planner_mode(self, mode):
        mode = str(mode).strip()
        if mode not in V71_PLANNER_MODES:
            raise ValueError(f"unsupported planner mode: {mode}; choose {V71_PLANNER_MODES}")
        self._planner_mode = mode
        self.emit(f"[V71 PLANNER SELECT] {mode}")
        return mode

    def gpu_planner_status(self, refresh=True):
        if self._gpu_client is None:
            self._last_gpu_status = {
                "ok": False, "available": False,
                "error": "NERO_GPU_Planner_Client unavailable in ROS runtime",
            }
            return dict(self._last_gpu_status)
        if refresh:
            self._last_gpu_status = self._gpu_client.ping(timeout_s=0.7)
        return dict(self._last_gpu_status)

    def _v71_pose_to_curobo(self, pose, local_offset=(0.0, 0.0, 0.0)):
        q = (
            float(pose.orientation.x), float(pose.orientation.y),
            float(pose.orientation.z), float(pose.orientation.w),
        )
        off = quat_rotate_vector(q, local_offset)
        return [
            float(pose.position.x) + off[0],
            float(pose.position.y) + off[1],
            float(pose.position.z) + off[2],
            float(pose.orientation.w), float(pose.orientation.x),
            float(pose.orientation.y), float(pose.orientation.z),
        ]

    def _v71_moveit_world_snapshot(self, timeout_s=0.8, cache_s=0.45):
        """Approximate MoveIt world objects as OBBs for cuRobo planning.

        Primitive boxes stay exact. Spheres/cylinders become conservative boxes.
        Meshes become oriented local-AABB boxes. MoveIt remains the final
        authoritative trajectory validity gate after GPU planning.
        """
        now = time.time()
        if self._v71_world_cache is not None and now - self._v71_world_cache_time < float(cache_s):
            return self._v71_world_cache
        if not self.planning_scene_client.wait_for_service(timeout_sec=0.25):
            out = {"ok": False, "source": "moveit", "cuboids": [], "error": "/get_planning_scene unavailable"}
            self._v71_world_cache = out; self._v71_world_cache_time = now
            return out
        req = GetPlanningScene.Request()
        req.components.components = int(PlanningSceneComponents.WORLD_OBJECT_GEOMETRY)
        try:
            resp = self.wait_future(
                self.planning_scene_client.call_async(req),
                float(timeout_s),
                "MoveIt planning scene",
            )
        except Exception as exc:
            out = {"ok": False, "source": "moveit", "cuboids": [], "error": str(exc)}
            self._v71_world_cache = out; self._v71_world_cache_time = now
            return out
        cuboids = []
        skipped = 0
        objects = list(getattr(getattr(resp.scene, "world", None), "collision_objects", []) or [])
        for obj_idx, obj in enumerate(objects):
            frame = str(getattr(getattr(obj, "header", None), "frame_id", "") or "base_link")
            if frame not in ("base_link", "world"):
                skipped += 1
                continue
            oid = str(getattr(obj, "id", f"obj{obj_idx}"))
            prims = list(getattr(obj, "primitives", []) or [])
            pposes = list(getattr(obj, "primitive_poses", []) or [])
            for i, (prim, pose) in enumerate(zip(prims, pposes)):
                dims = list(getattr(prim, "dimensions", []) or [])
                ptype = int(getattr(prim, "type", 0))
                box_dims = None
                if ptype == int(SolidPrimitive.BOX) and len(dims) >= 3:
                    box_dims = [float(dims[0]), float(dims[1]), float(dims[2])]
                elif ptype == int(SolidPrimitive.SPHERE) and len(dims) >= 1:
                    r = float(dims[0]); box_dims = [2*r, 2*r, 2*r]
                elif ptype == int(SolidPrimitive.CYLINDER) and len(dims) >= 2:
                    h = float(dims[0]); r = float(dims[1]); box_dims = [2*r, 2*r, h]
                elif ptype == int(SolidPrimitive.CONE) and len(dims) >= 2:
                    h = float(dims[0]); r = float(dims[1]); box_dims = [2*r, 2*r, h]
                if box_dims and min(box_dims) > 1e-6:
                    cuboids.append({
                        "name": f"{oid}_prim{i}",
                        "pose": self._v71_pose_to_curobo(pose),
                        "dims": box_dims,
                    })
            meshes = list(getattr(obj, "meshes", []) or [])
            mposes = list(getattr(obj, "mesh_poses", []) or [])
            # Mesh->OBB can incorrectly fill the hollow interior of baskets/holders.
            # Keep it opt-in; MoveIt remains the final authority for mesh collisions.
            mesh_obb = os.environ.get("NERO_GPU_WORLD_MESH_OBB", "0") == "1"
            if not mesh_obb:
                skipped += len(meshes)
            else:
                for i, (mesh, pose) in enumerate(zip(meshes, mposes)):
                    verts = list(getattr(mesh, "vertices", []) or [])
                    if not verts:
                        skipped += 1
                        continue
                    xs = [float(v.x) for v in verts]; ys = [float(v.y) for v in verts]; zs = [float(v.z) for v in verts]
                    lo = (min(xs), min(ys), min(zs)); hi = (max(xs), max(ys), max(zs))
                    dims = [hi[k] - lo[k] for k in range(3)]
                    if min(dims) <= 1e-6:
                        skipped += 1
                        continue
                    center = tuple(0.5 * (lo[k] + hi[k]) for k in range(3))
                    cuboids.append({
                        "name": f"{oid}_mesh{i}",
                        "pose": self._v71_pose_to_curobo(pose, center),
                        "dims": dims,
                    })
        out = {"ok": True, "source": "moveit", "cuboids": cuboids, "skipped": skipped, "objects": len(objects)}
        self._v71_world_cache = out; self._v71_world_cache_time = now
        return out

    def _v71_full_joint_state_for_arm(self, arm_positions):
        live = self._current_joint_state_copy()
        out = JointState()
        out.header = live.header
        out.name = list(live.name)
        out.position = list(live.position)
        out.velocity = list(live.velocity)
        out.effort = list(live.effort)
        name_to_i = {str(n): i for i, n in enumerate(out.name)}
        for name, value in zip(ARM_JOINT_NAMES, arm_positions):
            if name in name_to_i:
                out.position[name_to_i[name]] = float(value)
            else:
                out.name.append(name)
                out.position.append(float(value))
        return out

    def _v71_state_is_valid(self, arm_positions, timeout_s=1.0):
        if not self.state_validity_client.wait_for_service(timeout_sec=0.20):
            return False, "MoveIt /check_state_validity unavailable"
        req = GetStateValidity.Request()
        req.robot_state = RobotState()
        req.robot_state.joint_state = self._v71_full_joint_state_for_arm(arm_positions)
        req.robot_state.is_diff = False
        req.group_name = "arm"
        try:
            resp = self.wait_future(
                self.state_validity_client.call_async(req),
                float(timeout_s),
                "MoveIt state validity",
            )
        except Exception as exc:
            return False, f"state-validity exception: {exc}"
        if resp is None:
            return False, "state-validity returned no response"
        if bool(resp.valid):
            return True, "OK"
        contacts = list(getattr(resp, "contacts", []) or [])
        if contacts:
            c = contacts[0]
            b1 = getattr(c, "contact_body_1", "?")
            b2 = getattr(c, "contact_body_2", "?")
            return False, f"collision:{b1}<->{b2}"
        return False, "MoveIt state invalid (collision/constraint)"

    def _v71_validate_gpu_positions(self, positions):
        if not positions or len(positions) < 2:
            return False, "GPU trajectory has fewer than 2 points"
        if len(positions) > V71_GPU_MAX_POINTS:
            return False, f"GPU trajectory too dense: {len(positions)}>{V71_GPU_MAX_POINTS}"
        for row_idx, q in enumerate(positions):
            if len(q) != 7:
                return False, f"GPU trajectory row {row_idx} is not 7-DOF"
            for j, (v, lim) in enumerate(zip(q, JOINT_LIMITS_RAD)):
                if not math.isfinite(float(v)):
                    return False, f"GPU trajectory row {row_idx} J{j+1} non-finite"
                if float(v) < float(lim[0]) - 1e-5 or float(v) > float(lim[1]) + 1e-5:
                    return False, (
                        f"GPU trajectory row {row_idx} J{j+1} hard-limit violation "
                        f"{math.degrees(float(v)):.2f}deg"
                    )

        # V2-23: MoveIt is the authoritative environment-collision gate.
        # Check EVERY interpolated cuRobo point, never a sampled subset.
        n = len(positions)
        for idx in range(n):
            ok, reason = self._v71_state_is_valid(positions[idx])
            if not ok:
                return False, f"MoveIt revalidation failed at GPU point {idx}/{n-1}: {reason}"
        return True, f"MoveIt revalidation PASS ({n}/{n} points)"

    def _v71_gpu_response_to_trajectory(self, response, velocity_scale, accel_scale):
        positions = response.get("positions") or []
        velocities = response.get("velocities") or []
        times = response.get("times") or []
        names = response.get("joint_names") or []
        if list(names) != list(ARM_JOINT_NAMES):
            raise RuntimeError(f"GPU joint order mismatch: {names}")
        ok, reason = self._v71_validate_gpu_positions(positions)
        if not ok:
            raise RuntimeError(reason)
        self.emit(f"[V71 GPU SAFETY GATE] {reason}")

        n = len(positions)
        if len(times) != n:
            # Fall back to a conservative synthetic 50 Hz clock before scaling.
            times = [0.02 * i for i in range(n)]
        vel_scale = max(0.03, min(1.0, float(velocity_scale)))
        acc_scale = max(0.03, min(1.0, float(accel_scale)))
        dilation = max(1.0 / vel_scale, 1.0 / math.sqrt(acc_scale))

        traj = RobotTrajectory()
        traj.joint_trajectory.joint_names = list(ARM_JOINT_NAMES)
        last_t = -1.0
        for i, q in enumerate(positions):
            p = JointTrajectoryPoint()
            p.positions = [float(v) for v in q]
            if i < len(velocities) and len(velocities[i]) == 7:
                p.velocities = [float(v) / dilation for v in velocities[i]]
            t = max(float(times[i]) * dilation, (last_t + 0.005) if last_t >= 0.0 else 0.0)
            p.time_from_start = Duration(seconds=t).to_msg()
            last_t = t
            traj.joint_trajectory.points.append(p)
        return traj

    def _v71_try_gpu_pose(self, xyz, quat_xyzw, velocity_scale, accel_scale):
        """V2-23 GPU pose planner.

        cuRobo does NOT receive the environment world.  MoveIt remains the single
        authoritative environment-collision model.  If MoveIt rejects a cuRobo
        trajectory, request a fresh cuRobo trajectory and try again, up to at
        least 10 outer replans.  Strict mode never falls back silently.
        """
        if self._gpu_client is None:
            raise RuntimeError("GPU planner client module is unavailable")
        mode = str(self._planner_mode)
        engine = {
            "GPU Auto": "auto",
            "cuMotion": "cumotion",
            "cuMotion Strict": "cumotion",
            "cuRobo": "curobo",
            "cuRobo Strict": "curobo",
        }.get(mode)
        if engine is None:
            raise RuntimeError("GPU planner is disabled by GUI mode")

        live = self._current_joint_state_copy()
        q_start = self._arm_positions_from_joint_state(live)
        retry_budget = V71_GPU_REPLAN_ATTEMPTS if engine == "curobo" else 1
        failures = []
        # V3.03 FAILURE CLASSIFICATION: "plan_pose returned None" means the
        # goal pose itself has no IK solution; a new random seed cannot fix
        # that.  Stop after N consecutive None results and tell the caller to
        # change the GOAL (yaw / height / slot) instead of the seed.
        none_abort303 = max(1, int(os.environ.get("NERO_V303_NONE_ABORT", "3")))
        none_run303 = 0

        self.emit(
            f"[V2-23 GPU REQUEST] mode={mode} | world-sync=OFF | "
            f"MoveIt full-trajectory gate=ON | replan_budget={retry_budget} | XYZ "
            f"[{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}]"
        )

        for attempt in range(1, retry_budget + 1):
            seed = 23000 + attempt
            self.emit(
                f"[V2-23 CUROBO ATTEMPT] {attempt}/{retry_budget} | "
                f"seed={seed} | world=OFF"
            )
            try:
                response = self._gpu_client.plan_pose(
                    engine=engine,
                    joint_names=ARM_JOINT_NAMES,
                    q_start=q_start,
                    position=xyz,
                    quat_xyzw=quat_normalize(quat_xyzw),
                    velocity_scale=velocity_scale,
                    accel_scale=accel_scale,
                    world=None,
                    max_attempts=5,
                    retry_index=attempt,
                    random_seed=seed,
                    timeout_s=30.0,
                )
            except Exception as exc:
                reason = f"request exception: {exc}"
                failures.append((attempt, reason))
                self.emit(f"[V2-23 CUROBO REPLAN] attempt={attempt}/{retry_budget} | {reason}")
                continue

            if not bool(response.get("ok")):
                reason = str(response.get("error", "GPU planner returned failure"))
                failures.append((attempt, reason))
                self.emit(f"[V2-23 CUROBO REPLAN] attempt={attempt}/{retry_budget} | planner={reason}")
                if "returned None" in reason:
                    none_run303 += 1
                    if none_run303 >= none_abort303:
                        self.emit(
                            f"[V3.03 GOAL UNREACHABLE] {none_run303} consecutive 'returned None' -> "
                            f"goal has no IK solution; stop seed retries, change the goal"
                        )
                        raise RuntimeError(
                            f"GOAL_IK_UNREACHABLE: cuRobo plan_pose returned None {none_run303}x in a row "
                            f"(goal pose has no IK solution); change yaw/height/slot, not the seed"
                        )
                else:
                    none_run303 = 0
                continue

            positions = response.get("positions") or []
            if positions and len(positions[0]) == 7:
                start_err = max(abs(float(positions[0][j]) - float(q_start[j])) for j in range(7))
                if start_err > math.radians(10.0):
                    reason = f"trajectory start discontinuity {math.degrees(start_err):.1f}deg > 10deg"
                    failures.append((attempt, reason))
                    self.emit(f"[V2-23 CUROBO REPLAN] attempt={attempt}/{retry_budget} | {reason}")
                    continue

            try:
                traj = self._v71_gpu_response_to_trajectory(response, velocity_scale, accel_scale)
            except Exception as exc:
                reason = str(exc)
                # Infrastructure failure is not solved by generating another path.
                if ("state-validity unavailable" in reason or
                    "state-validity exception" in reason or
                    "returned no response" in reason):
                    raise RuntimeError(reason)
                failures.append((attempt, reason))
                self.emit(
                    f"[V2-23 MOVEIT REJECT -> CUROBO REPLAN] "
                    f"attempt={attempt}/{retry_budget} | {reason}"
                )
                continue

            self.emit(
                f"[V2-23 GPU PLAN OK] engine={response.get('engine')} | "
                f"attempt={attempt}/{retry_budget} | "
                f"points={len(traj.joint_trajectory.points)} | "
                f"planner={float(response.get('planning_ms', 0.0)):.1f}ms | "
                f"raw_shape={response.get('raw_position_shape')} | world=OFF | "
                f"MoveIt=PASS"
            )
            return traj

        tail = "; ".join(f"#{i}:{r}" for i, r in failures[-5:])
        raise RuntimeError(
            f"cuRobo exhausted {retry_budget} trajectory attempts; "
            f"none passed MoveIt full-trajectory validation. tail=[{tail}]"
        )

    def _plan_pose(
        self,
        xyz,
        quat_xyzw,
        velocity_scale=0.25,
        accel_scale=0.25,
        orientation_tolerance=0.20,
        position_tolerance=0.010,
        planner_id="",
        planning_attempts=8,
        planning_time=8.0,
    ):
        """V71.3 / V2-23 dispatcher: GPU planner first; Strict modes never silently fall back.

        The mode affects free-space pose planning only.  Existing strict Cartesian
        approach/lift/insertion primitives and V2-15/V2-20 local recovery remain
        untouched in V2-22 so A/B testing cannot silently weaken their safety logic.
        """
        if str(self._planner_mode) != "BioIK/MoveIt":
            try:
                return self._v71_try_gpu_pose(
                    xyz, quat_xyzw, velocity_scale, accel_scale,
                )
            except Exception as exc:
                if str(self._planner_mode).endswith(" Strict"):
                    self.emit(f"[V71 GPU STRICT FAIL] mode={self._planner_mode} | {exc}")
                    raise
                self.emit(
                    f"[V71 GPU FALLBACK -> BIOIK] mode={self._planner_mode} | {exc}"
                )
        else:
            self.emit("[V71 BIOIK SELECTED] GUI/manual planner mode")

        return self._plan_pose_bioik_v70(
            xyz,
            quat_xyzw,
            velocity_scale=velocity_scale,
            accel_scale=accel_scale,
            orientation_tolerance=orientation_tolerance,
            position_tolerance=position_tolerance,
            planner_id=planner_id,
            planning_attempts=planning_attempts,
            planning_time=planning_time,
        )

    def _plan_pose_bioik_v70(
        self,
        xyz,
        quat_xyzw,
        velocity_scale=0.25,
        accel_scale=0.25,
        orientation_tolerance=0.20,
        position_tolerance=0.010,
        planner_id="",
        planning_attempts=8,
        planning_time=8.0,
    ):
        """V70 natural pose planning.

        Instead of allowing MoveIt to choose an arbitrary redundant IK endpoint,
        expose several collision-aware BioIK solutions, rank them by minimax joint
        travel + wrist reserve, plan the best few as joint goals, then choose the
        route with the least actual joint winding.  Legacy pose planning is retained
        only as a reliability fallback if the ranked pipeline cannot produce a plan.
        """
        self.emit(
            "[V70 NATURAL PLAN] pose -> ranked collision-aware IK -> joint-goal OMPL | "
            f"XYZ [{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}]"
        )
        candidates,ik_errors,current_arm=self._v70_ranked_pose_ik_candidates(
            xyz,quat_xyzw,max_candidates=V70_POSE_IK_TOP_K,timeout_s=0.14,
        )
        planned=[]
        for endpoint_score,joints,seed_tag,endpoint_metrics in candidates:
            try:
                traj=self._plan_joint_solution(
                    joints,
                    velocity_scale=velocity_scale,
                    accel_scale=accel_scale,
                    planner_id=planner_id,
                    planning_attempts=planning_attempts,
                    planning_time=planning_time,
                )
            except Exception as exc:
                self.emit(f"[V70 ROUTE REJECT] seed={seed_tag} | joint-goal plan failed: {exc}")
                continue
            tm=self._v70_trajectory_metrics(traj,current_arm,joints)
            route_score=(
                float(tm['max_cumulative_norm']),
                float(tm['max_extra_norm']),
                float(tm['total_cumulative_norm']),
            )+tuple(endpoint_score)
            planned.append((route_score,traj,joints,seed_tag,endpoint_metrics,tm))
            self.emit(
                f"[V70 ROUTE CANDIDATE] seed={seed_tag} | "
                f"max_path_joint={max(tm['cumulative_deg']):.1f}deg | "
                f"path_total={sum(tm['cumulative_deg']):.1f}deg | "
                f"max_extra_norm={tm['max_extra_norm']:.3f}"
            )

        if planned:
            planned.sort(key=lambda item:item[0])
            score,traj,joints,seed_tag,em,tm=planned[0]
            self.emit(
                f"[V70 NATURAL SELECT] seed={seed_tag} | "
                f"max_endpoint_dJ={max(em['delta_deg']):.1f}deg | "
                f"endpoint_total={em['sum_delta_deg']:.1f}deg | "
                f"wrist_reserve={em['min_wrist_reserve_deg']:.1f}deg | "
                f"planned_max_joint_travel={max(tm['cumulative_deg']):.1f}deg"
            )
            return traj

        # Reliability fallback: keep V69's original collision-checked pose planning
        # if BioIK multi-candidate enumeration happens to fail in an unusual state.
        self.emit(
            f"[V70 LEGACY POSE FALLBACK] ranked IK/joint-goal planning produced no route | "
            f"ik_tail={ik_errors[-3:]}"
        )
        if not self.move_client.wait_for_server(timeout_sec=3.0):
            raise RuntimeError("/move_action unavailable")
        goal=self._build_move_group_goal(
            xyz,quat_xyzw,velocity_scale,accel_scale,
            orientation_tolerance,position_tolerance,
            planner_id=planner_id,planning_attempts=planning_attempts,
            planning_time=planning_time,
        )
        gh=self.wait_future(self.move_client.send_goal_async(goal),8.0,"MoveGroup goal response")
        if gh is None or not gh.accepted:
            raise RuntimeError("MoveGroup rejected planning goal")
        wrapped=self.wait_future(gh.get_result_async(),20.0,"MoveGroup planning")
        if wrapped is None:
            raise RuntimeError("MoveGroup returned no planning result")
        result=wrapped.result
        if int(result.error_code.val)!=1:
            raise RuntimeError("PLANNING FAILED: "+moveit_error_text(result.error_code))
        traj=result.planned_trajectory
        n=len(traj.joint_trajectory.points)
        if n==0:
            self.emit("MoveIt: goal already satisfied")
            return None
        self.emit(f"Plan OK (legacy fallback): {n} points | {float(result.planning_time):.3f}s")
        return traj

    def _plan_pose_with_orientation_path_constraint(
        self,
        xyz,
        quat_xyzw,
        velocity_scale=0.18,
        accel_scale=0.18,
        orientation_tolerance=0.16,
        path_orientation_tolerance=0.22,
        position_tolerance=0.012,
    ):
        """OMPL fallback that keeps the carried object's grasp orientation.

        This is used only after local Cartesian corridor candidates fail.
        OMPL is then allowed to bend the arm around obstacles while a TCP
        orientation path constraint prevents arbitrary wrist flipping.
        """
        if not self.move_client.wait_for_server(timeout_sec=3.0):
            raise RuntimeError("/move_action unavailable")

        goal = self._build_move_group_goal(
            xyz,
            quat_xyzw,
            velocity_scale,
            accel_scale,
            orientation_tolerance,
            position_tolerance,
        )

        q = quat_normalize(quat_xyzw)
        path_c = Constraints()
        oc = OrientationConstraint()
        oc.header.frame_id = "base_link"
        oc.link_name = "tcp_link"
        oc.orientation.x = float(q[0])
        oc.orientation.y = float(q[1])
        oc.orientation.z = float(q[2])
        oc.orientation.w = float(q[3])
        oc.absolute_x_axis_tolerance = float(path_orientation_tolerance)
        oc.absolute_y_axis_tolerance = float(path_orientation_tolerance)
        oc.absolute_z_axis_tolerance = float(path_orientation_tolerance)
        oc.weight = 1.0
        path_c.orientation_constraints = [oc]
        goal.request.path_constraints = path_c
        goal.request.num_planning_attempts = 14
        goal.request.allowed_planning_time = 8.0

        self.emit(
            "[ADAPTIVE OMPL] local Cartesian candidates exhausted; "
            f"planning obstacle-avoiding path to [{xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}] "
            "with grasp-orientation path constraint"
        )

        gh = self.wait_future(
            self.move_client.send_goal_async(goal),
            8.0,
            "Adaptive OMPL goal response",
        )
        if gh is None or not gh.accepted:
            raise RuntimeError("Adaptive OMPL rejected planning goal")

        wrapped = self.wait_future(
            gh.get_result_async(),
            25.0,
            "Adaptive OMPL planning",
        )
        if wrapped is None:
            raise RuntimeError("Adaptive OMPL returned no planning result")

        result = wrapped.result
        if int(result.error_code.val) != 1:
            raise RuntimeError(
                "ADAPTIVE OMPL FAILED: " + moveit_error_text(result.error_code)
            )

        traj = result.planned_trajectory
        n = len(traj.joint_trajectory.points)
        if n == 0:
            self.emit("[ADAPTIVE OMPL] target already satisfied")
            return None

        self.emit(
            f"[ADAPTIVE OMPL OK] {n} points | "
            f"{float(result.planning_time):.3f}s"
        )
        return traj

    def _execute_trajectory(self, robot_trajectory):
        """Execute through arm_controller with *physical* safety acknowledgement.

        V59 fixes a critical false-positive from V59: the ros2_control FakeSystem
        could finish a trajectory after MuJoCo's safety guard had already stopped
        accepting commands.  The controller then returned SUCCESS although the
        physical TCP was nowhere near the goal.

        V59 therefore:
          1) watches the MuJoCo safety-trip counter during execution;
          2) requests a non-destructive soft recover if a trip occurs;
          3) verifies real /feedback/joint_states against the final trajectory.
        """
        if robot_trajectory is None:
            return True

        if not self.arm_controller_client.wait_for_server(timeout_sec=3.0):
            raise RuntimeError(
                "/arm_controller/follow_joint_trajectory unavailable"
            )

        safety0 = self._safety_trip_count()

        # V3.05: never drive the arm through an ambiguous controller graph.
        self._v305_check_single_arm_controller()

        # While gate is OFF, align the mock controller with the *physical* robot.
        self._set_arm_command_gate(False)
        self._sync_arm_controller_to_feedback()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = robot_trajectory.joint_trajectory

        self.emit(
            f"Executing planned trajectory; ROS arm gate ON | safety baseline={safety0}"
        )
        self._set_arm_command_gate(True)

        gh = None
        try:
            gh = self.wait_future(
                self.arm_controller_client.send_goal_async(goal),
                6.0,
                "arm_controller goal response",
            )
            if gh is None or not gh.accepted:
                raise RuntimeError("EXECUTION_LAYER_FAULT: arm_controller rejected trajectory")

            with self._state_lock:
                self._active_goal_handle = gh

            result_future = gh.get_result_async()
            # V3.01 DURATION-AWARE EXECUTION TIMEOUT
            # The fixed 45.0 s deadline was shorter than some valid trajectories
            # (e.g. slow q0 fallback: retime 27.8 s x slow_factor 1.65 = 45.8 s),
            # so they were cancelled ~1 s before completion.  Deadline is now
            # max(45 s, 1.15 x planned duration + 15 s), capped at 180 s.
            try:
                _pts301 = list(goal.trajectory.points)
                _tf301 = _pts301[-1].time_from_start if _pts301 else None
                traj_dur301 = (float(_tf301.sec) + float(_tf301.nanosec) * 1e-9) if _tf301 is not None else 0.0
            except Exception:
                traj_dur301 = 0.0
            timeout_s301 = min(180.0, max(45.0, 1.15 * traj_dur301 + 15.0))
            if timeout_s301 > 45.0:
                self.emit(
                    f"[V3.01 EXEC TIMEOUT] planned_duration={traj_dur301:.1f}s -> deadline={timeout_s301:.1f}s"
                )
            deadline = time.time() + timeout_s301
            wrapped = None

            while time.time() < deadline:
                safety_now = self._safety_trip_count()
                if safety_now != safety0:
                    try:
                        gh.cancel_goal_async()
                    except Exception:
                        pass
                    self._set_arm_command_gate(False)
                    self.emit(
                        f"[PHYSICAL SAFETY STOP] trip {safety0}->{safety_now}; "
                        "controller cancelled; arm frozen in place; gripper preserved; "
                        "SAFETY LATCH HELD (press STOP + RECOVER to unlatch)"
                    )
                    raise RuntimeError(
                        f"PHYSICAL_SAFETY_STOP trip {safety0}->{safety_now}"
                    )

                worker_epoch = getattr(self._worker_local, "epoch", None)
                if worker_epoch is not None:
                    with self._state_lock:
                        current_epoch = int(self._recovery_epoch)
                    if int(worker_epoch) != current_epoch:
                        raise RuntimeError("skill aborted by recovery/reset")

                if result_future.done():
                    wrapped = result_future.result()
                    break
                if self._shutdown:
                    raise RuntimeError("Skill core shutting down")
                time.sleep(0.01)

            if wrapped is None:
                # V2-51 TIMEOUT CANCEL BEFORE HANDOFF
                # Keep the original 45.0 s deadline.  On timeout, explicitly cancel
                # the stale controller goal before the backend considers a safe-high
                # current-state replan.  Do NOT extend the old trajectory.
                try:
                    cancel_future51 = gh.cancel_goal_async()
                    try:
                        self.wait_future(
                            cancel_future51, 2.0,
                            "arm_controller V2-51 timeout cancel",
                        )
                    except Exception as cancel_wait_exc51:
                        self.emit(
                            f"[V2-51 TIMEOUT CANCEL WARN] cancel acknowledgement not confirmed: {cancel_wait_exc51}"
                        )
                except Exception as cancel_exc51:
                    self.emit(
                        f"[V2-51 TIMEOUT CANCEL WARN] cancel request failed: {cancel_exc51}"
                    )
                self._set_arm_command_gate(False)
                self.emit(
                    f"[V2-51 TIMEOUT CANCEL] {timeout_s301:.1f}s deadline reached; old trajectory cancelled/gated OFF before any recovery handoff"
                )
                raise RuntimeError(f"arm_controller execution timeout after {timeout_s301:.1f}s")

            result = wrapped.result
            code = int(result.error_code)
            if code != 0:
                name = FJT_ERROR_NAMES.get(code, f"UNKNOWN_{code}")
                msg = getattr(result, "error_string", "")
                raise RuntimeError(
                    f"ARM EXECUTION FAILED: {name} ({code}) {msg}"
                )
            # V3.05: error_code==0 is NOT enough (cancelled / pre-empted goals carry 0 too).
            self._v305_require_succeeded(wrapped, "trajectory")

            # Race-proof: the safety trip may arrive immediately after the action result.
            safety_now = self._safety_trip_count()
            if safety_now != safety0:
                self.emit(
                    f"[SAFETY LATCH HELD] trip {safety0}->{safety_now}; "
                    "automatic unlatch disabled"
                )
                raise RuntimeError(
                    f"PHYSICAL_SAFETY_STOP trip {safety0}->{safety_now}"
                )

            self._verify_physical_joint_target(robot_trajectory)
            self.emit("Arm trajectory complete + physical feedback verified")
            return True

        finally:
            self._set_arm_command_gate(False)
            self.emit("ROS arm gate OFF; manual controls own idle robot")

    def move_to_pose(
        self,
        xyz,
        quat_xyzw=None,
        velocity_scale=0.25,
        accel_scale=0.25,
        orientation_tolerance=0.20,
        position_tolerance=0.010,
    ):
        if quat_xyzw is None:
            _, quat_xyzw = self.current_tcp_pose()
        traj = self._plan_pose(
            xyz,
            quat_xyzw,
            velocity_scale,
            accel_scale,
            orientation_tolerance,
            position_tolerance,
        )
        return self._execute_trajectory(traj)

    def move_to(self, x, y, z):
        _, q = self.current_tcp_pose()
        return self.move_to_pose([x, y, z], q)

    def lift(self, dz=0.10):
        """
        V53 safe lift: strict vertical Cartesian retreat on the CURRENT IK
        branch. The old pose-goal lift allowed redundant 7-DOF IK to choose a
        distant joint solution, which could flip the wrist/arm even though TCP
        start/end poses looked similar.
        """
        dz = float(dz)
        if dz <= 0.0:
            raise ValueError("lift distance must be > 0")

        start_xyz, current_q = self.current_tcp_pose()
        q = self.active_grasp_quat or current_q

        # Explicit short vertical waypoints keep interpolation on the local
        # redundancy branch and make failures stop cleanly instead of falling
        # back to a global pose planner.
        n = max(2, int(math.ceil(dz / 0.025)))
        waypoints = []
        for k in range(1, n + 1):
            f = float(k) / float(n)
            waypoints.append([
                float(start_xyz[0]),
                float(start_xyz[1]),
                float(start_xyz[2]) + dz * f,
            ])

        self.emit(
            f"[LIFT] strict vertical Cartesian +{dz*1000.0:.0f} mm | "
            f"{n} waypoint(s) | fixed grasp orientation"
        )

        traj = self._cartesian_waypoint_plan(
            waypoints,
            q,
            label=f"vertical lift +{dz*1000.0:.0f}mm",
            max_step=0.003,
            max_speed=0.045,
        )
        self._execute_trajectory(traj)

        verify_deadline = time.time() + 2.5
        gained = -1e9
        while time.time() < verify_deadline:
            end_xyz, _ = self.current_tcp_pose()
            gained = float(end_xyz[2]) - float(start_xyz[2])
            if gained >= max(0.010, 0.65 * dz):
                break
            time.sleep(0.05)

        self.emit(
            f"[LIFT VERIFY] requested {dz*1000.0:.0f} mm | "
            f"TCP gained {gained*1000.0:.1f} mm"
        )
        if gained < max(0.010, 0.65 * dz):
            raise RuntimeError(
                f"lift verification failed: TCP rose only {gained*1000.0:.1f} mm"
            )

        self.active_grasp_quat = quat_normalize(q)
        self.emit("lift complete | straight Cartesian retreat")
        return True


    def _arm_positions_from_joint_state(self, js):
        name_to_pos = {
            str(n): float(p)
            for n, p in zip(js.name, js.position)
        }
        if not all(name in name_to_pos for name in ARM_JOINT_NAMES):
            missing = [name for name in ARM_JOINT_NAMES if name not in name_to_pos]
            raise RuntimeError(
                "Current feedback missing arm joints: " + ",".join(missing)
            )
        return [name_to_pos[name] for name in ARM_JOINT_NAMES]

    def _seed_with_arm_positions(self, template_js, arm_positions):
        """Return a complete JointState seed with J1..J7 replaced.

        TRAC-IK gets the previous accepted solution as its next seed.  This is
        the key continuation mechanism that keeps a redundant 7-DOF arm on the
        same local IK branch instead of jumping to a distant elbow/wrist pose.
        """
        out = JointState()
        out.header = template_js.header
        out.header.stamp = self.node.get_clock().now().to_msg()
        out.name = list(template_js.name)
        out.position = list(template_js.position)
        out.velocity = list(template_js.velocity)
        out.effort = list(template_js.effort)

        index = {str(n): i for i, n in enumerate(out.name)}
        for name, q in zip(ARM_JOINT_NAMES, arm_positions):
            if name not in index:
                raise RuntimeError(f"IK seed missing {name}")
            out.position[index[name]] = float(q)
        return out

    @staticmethod
    def _xyz_distance(a, b):
        return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))

    @staticmethod
    def _max_joint_delta(a, b):
        return max(abs(float(a[i]) - float(b[i])) for i in range(7))

    def _make_incremental_joint_trajectory(self, arm_points, xyz_points):
        """Build one dynamics-safer FollowJointTrajectory from validated IK samples.

        V70 fixes the V65 timing defect: timing is no longer based only on TCP
        distance.  Every segment is retimed against BOTH Cartesian travel and
        joint-space travel, then a forward/backward acceleration pass stretches
        neighboring segments until predicted joint acceleration is conservative.

        The resulting point velocities are continuous/monotone where possible,
        so JointTrajectoryController receives a smooth cubic trajectory instead
        of tiny Cartesian waypoints with potentially aggressive hidden joint
        motion.
        """
        if len(arm_points) != len(xyz_points):
            raise RuntimeError("incremental trajectory point count mismatch")
        if len(arm_points) < 2:
            return None

        n = len(arm_points)
        q = [[float(v) for v in p] for p in arm_points]
        xyz = [[float(v) for v in p] for p in xyz_points]

        # V70: speed up the now-stable V66 carried-object execution while staying
        # below the V31 normal reference envelope (~0.35-0.40 rad/s).  The
        # joint-aware acceleration smoothing and safety latch remain unchanged.
        joint_vel_limit = [0.28, 0.28, 0.28, 0.28, 0.32, 0.32, 0.32]  # rad/s
        joint_acc_limit = [0.75, 0.75, 0.75, 0.75, 0.85, 0.85, 0.85]  # rad/s^2
        tcp_speed_limit = 0.040  # 40 mm/s while carrying
        min_segment_dt = 0.22

        dq_segments = []
        dt = []
        for i in range(1, n):
            dq = [q[i][j] - q[i - 1][j] for j in range(7)]
            dq_segments.append(dq)
            dxyz = self._xyz_distance(xyz[i - 1], xyz[i])

            dt_tcp = dxyz / tcp_speed_limit if dxyz > 0.0 else 0.0
            dt_joint = max(
                abs(dq[j]) / joint_vel_limit[j]
                for j in range(7)
            )
            # Rest-to-rest lower bound for a short segment.  It is conservative
            # even though interior points will later receive nonzero velocities.
            dt_acc = max(
                math.sqrt(2.0 * abs(dq[j]) / joint_acc_limit[j])
                if abs(dq[j]) > 1e-12 else 0.0
                for j in range(7)
            )
            dt.append(max(min_segment_dt, dt_tcp, dt_joint, 1.15 * dt_acc))

        # Stretch adjacent segments until changes in segment velocity stay under
        # the conservative acceleration envelope.  This is a lightweight
        # joint-aware retimer suitable for our Python execution bridge; it avoids
        # pretending that TCP distance alone determines safe timing.
        for _ in range(20):
            changed = False
            vseg = [
                [dq_segments[i][j] / dt[i] for j in range(7)]
                for i in range(len(dt))
            ]
            for i in range(1, len(dt)):
                avg_dt = max(1e-6, 0.5 * (dt[i - 1] + dt[i]))
                ratio = 1.0
                for j in range(7):
                    a = abs(vseg[i][j] - vseg[i - 1][j]) / avg_dt
                    ratio = max(ratio, a / joint_acc_limit[j])
                if ratio > 1.0 + 1e-6:
                    scale = min(2.0, math.sqrt(ratio) * 1.08)
                    dt[i - 1] *= scale
                    dt[i] *= scale
                    changed = True
            if not changed:
                break

        vseg = [
            [dq_segments[i][j] / dt[i] for j in range(7)]
            for i in range(len(dt))
        ]

        # Point velocities for cubic interpolation.  At a direction reversal,
        # force zero velocity to avoid overshoot.  Endpoints are always zero.
        point_vel = [[0.0] * 7 for _ in range(n)]
        for i in range(1, n - 1):
            for j in range(7):
                vp = vseg[i - 1][j]
                vn = vseg[i][j]
                if vp * vn <= 0.0:
                    vel = 0.0
                else:
                    # Time-weighted average of neighboring segment slopes.
                    vel = (vp * dt[i] + vn * dt[i - 1]) / (dt[i - 1] + dt[i])
                lim = 0.80 * joint_vel_limit[j]
                point_vel[i][j] = max(-lim, min(lim, vel))

        # Diagnostics: predicted segment velocity / boundary acceleration.
        max_v_ratio = 0.0
        for i, vv in enumerate(vseg):
            for j in range(7):
                max_v_ratio = max(max_v_ratio, abs(vv[j]) / joint_vel_limit[j])
        max_a_ratio = 0.0
        for i in range(1, len(vseg)):
            avg_dt = max(1e-6, 0.5 * (dt[i - 1] + dt[i]))
            for j in range(7):
                a = abs(vseg[i][j] - vseg[i - 1][j]) / avg_dt
                max_a_ratio = max(max_a_ratio, a / joint_acc_limit[j])

        traj = RobotTrajectory()
        traj.joint_trajectory.joint_names = list(ARM_JOINT_NAMES)

        cumulative = 0.35
        for i in range(n):
            pt = JointTrajectoryPoint()
            pt.positions = list(q[i])
            pt.velocities = list(point_vel[i])
            if i > 0:
                cumulative += dt[i - 1]

            sec = int(cumulative)
            nsec = int(round((cumulative - sec) * 1_000_000_000))
            if nsec >= 1_000_000_000:
                sec += 1
                nsec -= 1_000_000_000
            pt.time_from_start.sec = sec
            pt.time_from_start.nanosec = nsec
            traj.joint_trajectory.points.append(pt)

        self.emit(
            f"[V70 RETIME] {n-1} segment(s) | duration={cumulative:.2f}s | "
            f"tcp_limit={tcp_speed_limit*1000.0:.0f}mm/s | "
            f"pred_v={max_v_ratio*100.0:.0f}% limit | "
            f"pred_a={max_a_ratio*100.0:.0f}% limit"
        )
        return traj

    def _try_incremental_ik_point(
        self,
        xyz,
        grasp_q,
        seed_js,
        previous_q,
        timeout_s=0.12,
        max_joint_step_deg=14.0,
        solver_mode="bio",
        allow_trac_fallback=True,
    ):
        """Solve one nearby IK point while preserving the local joint branch.

        V70 policy:
          - BioIK is the primary continuation solver.
          - TRAC-IK is a bounded fallback/reference solver.
          - A solution is accepted only if it is close to the previous q.
        """
        solver_mode = str(solver_mode).strip().lower()
        attempts = []
        if solver_mode == "trac":
            attempts = [("TRAC", self._compute_trac_ik)]
        else:
            attempts = [("BIO", self._compute_ik)]
            if allow_trac_fallback:
                attempts.append(("TRAC-FALLBACK", self._compute_trac_ik))

        last_err = "NO ATTEMPT"
        for solver_label, fn in attempts:
            joints, err = fn(
                xyz,
                grasp_q,
                timeout_s=float(timeout_s),
                avoid_collisions=True,
                seed_joint_state=seed_js,
            )
            if joints is None:
                last_err = f"{solver_label}: {err}"
                continue

            max_dq = self._max_joint_delta(previous_q, joints)
            if max_dq > math.radians(float(max_joint_step_deg)):
                last_err = (
                    f"{solver_label}: IK branch jump "
                    f"{math.degrees(max_dq):.1f}deg > "
                    f"{float(max_joint_step_deg):.1f}deg"
                )
                continue

            return joints, f"SUCCESS/{solver_label}"

        return None, last_err

    def _joint_limit_status(self, q):
        """Return deterministic joint-limit margins for the current local branch."""
        rows = []
        nearest = None
        for i, (value, limits) in enumerate(zip(q, JOINT_LIMITS_RAD), start=1):
            value = float(value)
            lo, hi = map(float, limits)
            d_lo = value - lo
            d_hi = hi - value
            if d_lo <= d_hi:
                side = "LOWER"
                margin = d_lo
                limit = lo
            else:
                side = "UPPER"
                margin = d_hi
                limit = hi
            row = {
                "joint": i,
                "q": value,
                "side": side,
                "limit": limit,
                "margin": margin,
            }
            rows.append(row)
            if nearest is None or margin < nearest["margin"]:
                nearest = row
        return nearest, rows

    @staticmethod
    def _joint_margin_text(rows):
        parts = []
        for r in rows:
            parts.append(
                f"J{r['joint']}={math.degrees(r['q']):+.1f}deg/"
                f"{r['side']} margin {math.degrees(r['margin']):.1f}deg"
            )
        return " | ".join(parts)

    def _joint_limit_status(self, q):
        """Return deterministic joint-limit margins for the current local branch."""
        rows = []
        nearest = None
        for i, (value, limits) in enumerate(zip(q, JOINT_LIMITS_RAD), start=1):
            value = float(value)
            lo, hi = map(float, limits)
            d_lo = value - lo
            d_hi = hi - value
            if d_lo <= d_hi:
                side = "LOWER"
                margin = d_lo
                limit = lo
            else:
                side = "UPPER"
                margin = d_hi
                limit = hi
            row = {
                "joint": i,
                "q": value,
                "side": side,
                "limit": limit,
                "margin": margin,
            }
            rows.append(row)
            if nearest is None or margin < nearest["margin"]:
                nearest = row
        return nearest, rows

    @staticmethod
    def _joint_margin_text(rows):
        parts = []
        for r in rows:
            parts.append(
                f"J{r['joint']}={math.degrees(r['q']):+.1f}deg/"
                f"{r['side']} margin {math.degrees(r['margin']):.1f}deg"
            )
        return " | ".join(parts)

    def _diagnose_incremental_failure(
        self,
        xyz,
        grasp_q,
        seed_js,
        previous_q,
        solver_mode="bio",
        max_joint_step_deg=14.0,
    ):
        """Classify local failure: collision, remote branch, joint limit, or reachability.

        The collision-OFF diagnostic mirrors the actual dual-solver architecture:
        BioIK mode probes BioIK first *and* TRAC-IK as a reference/fallback.  A
        nearby collision-OFF solution from either solver proves that kinematics
        are locally feasible and points to collision filtering as the blocker.

        If neither collision-OFF solver can find any IK, current hard-joint-limit
        margins are used to distinguish a likely joint-limit frontier from a
        broader reachability/singularity frontier.  ``JOINT_LIMIT_LIKELY`` is
        intentionally probabilistic; GetPositionIK alone cannot prove causality.
        """
        nearest, rows = self._joint_limit_status(previous_q)
        nearest_margin_deg = math.degrees(float(nearest["margin"]))
        margin_text = self._joint_margin_text(rows)
        self.emit(
            f"[V70 JOINT MARGIN] nearest=J{nearest['joint']} {nearest['side']} "
            f"margin={nearest_margin_deg:.1f}deg | {margin_text}"
        )

        mode = str(solver_mode).lower()
        if mode == "trac":
            probes = [("TRAC", self._compute_trac_ik)]
        else:
            probes = [("BIO", self._compute_ik), ("TRAC", self._compute_trac_ik)]

        local_free = []
        remote_free = []
        noik = []
        for label, fn in probes:
            joints_free, err_free = fn(
                xyz,
                grasp_q,
                timeout_s=0.10,
                avoid_collisions=False,
                seed_joint_state=seed_js,
            )
            if joints_free is None:
                noik.append(f"{label}:{err_free}")
                continue

            max_dq = self._max_joint_delta(previous_q, joints_free)
            if max_dq <= math.radians(float(max_joint_step_deg)):
                local_free.append((label, max_dq))
            else:
                remote_free.append((label, max_dq))

        if local_free:
            label, max_dq = min(local_free, key=lambda x: x[1])
            reason = (
                "COLLISION/SELF-COLLISION: collision-OFF local IK exists via "
                f"{label} (max_dq={math.degrees(max_dq):.1f}deg)"
            )
            self.emit(f"[V70 FAILURE CLASS] COLLISION | {reason}")
            return reason

        if remote_free:
            label, max_dq = min(remote_free, key=lambda x: x[1])
            reason = (
                "LOCAL BRANCH ENDED: collision-OFF IK exists only on remote "
                f"branch via {label} (jump {math.degrees(max_dq):.1f}deg); "
                f"nearest current joint-limit margin={nearest_margin_deg:.1f}deg "
                f"at J{nearest['joint']}"
            )
            self.emit(f"[V70 FAILURE CLASS] REMOTE_BRANCH_ONLY | {reason}")
            return reason

        noik_text = " | ".join(noik) if noik else "no collision-OFF IK result"
        if nearest_margin_deg <= float(JOINT_LIMIT_LIKELY_MARGIN_DEG):
            reason = (
                f"JOINT-LIMIT FRONTIER (LIKELY): J{nearest['joint']}="
                f"{math.degrees(nearest['q']):+.1f}deg near {nearest['side']} "
                f"limit {math.degrees(nearest['limit']):+.1f}deg "
                f"(margin {nearest_margin_deg:.1f}deg); collision-OFF IK failed: "
                f"{noik_text}"
            )
            self.emit(f"[V70 FAILURE CLASS] JOINT_LIMIT_LIKELY | {reason}")
            return reason

        reason = (
            "KINEMATIC/REACHABILITY FRONTIER: collision-OFF IK failed in all "
            f"diagnostic solvers ({noik_text}); nearest hard joint-limit margin "
            f"is {nearest_margin_deg:.1f}deg at J{nearest['joint']}"
        )
        self.emit(f"[V70 FAILURE CLASS] REACHABILITY_OR_SINGULARITY | {reason}")
        return reason

    def _try_orientation_relaxation(
        self,
        xyz,
        base_grasp_q,
        current_tilt_deg,
        seed_js,
        previous_q,
        solver_mode="bio",
        allow_trac_fallback=True,
        max_joint_step_deg=14.0,
    ):
        """Search a small orientation cone while staying on the local joint branch.

        This is deliberately a bounded continuation search, not a free 6D pose
        planner.  Position stays fixed at the next 2.5 mm continuation point;
        yaw stays fixed; only world-frame roll/pitch may relax.  Candidate
        solutions are ranked by joint continuity, not by the first IK result.
        """
        cr, cp = map(float, current_tilt_deg)
        last_err = "NO ORIENTATION CANDIDATE"
        seen = set()

        # The search radius is measured in the roll/pitch tilt plane.  The arm
        # starts at (0,0) and is never allowed beyond a 10 degree total tilt.
        for shell_deg in (3.0, 6.0, 10.0):
            d = float(shell_deg)
            dd = d / math.sqrt(2.0)
            directions = (
                (+d, 0.0), (-d, 0.0), (0.0, +d), (0.0, -d),
                (+dd, +dd), (+dd, -dd), (-dd, +dd), (-dd, -dd),
            )
            valid = []

            self.emit(
                f"[V70 ORIENT SEARCH] XYZ=[{xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}] "
                f"| shell={shell_deg:.0f}deg | current_tilt=({cr:+.1f},{cp:+.1f})deg"
            )

            for dr, dp in directions:
                nr, np_ = cr + dr, cp + dp
                tilt_norm = math.hypot(nr, np_)
                if tilt_norm > 10.0 + 1e-9:
                    continue
                key = (round(nr, 3), round(np_, 3))
                if key in seen:
                    continue
                seen.add(key)

                q_candidate = quat_with_world_tilt(base_grasp_q, nr, np_)
                joints, status = self._try_incremental_ik_point(
                    xyz,
                    q_candidate,
                    seed_js,
                    previous_q,
                    timeout_s=0.16,
                    max_joint_step_deg=float(max_joint_step_deg),
                    solver_mode=solver_mode,
                    allow_trac_fallback=bool(allow_trac_fallback),
                )
                if joints is None:
                    last_err = str(status)
                    continue

                dq = [abs(float(joints[i]) - float(previous_q[i])) for i in range(7)]
                max_dq = max(dq)
                l2 = math.sqrt(sum(v*v for v in dq))
                natural_score, natural_metrics = self._v70_endpoint_score(previous_q, joints)
                # V70 keeps branch locality first, then prefers wrist reserve and
                # lower normalized joint travel among equally local candidates.
                score = (max_dq,) + tuple(natural_score) + (l2, tilt_norm)
                valid.append((score, joints, q_candidate, (nr, np_), status))

            if valid:
                valid.sort(key=lambda item: item[0])
                score, joints, q_candidate, tilt_deg, status = valid[0]
                self.emit(
                    f"[V70 ORIENT OK] tilt=(roll {tilt_deg[0]:+.1f}, "
                    f"pitch {tilt_deg[1]:+.1f})deg | "
                    f"max_dq={math.degrees(score[0]):.1f}deg | {status}"
                )
                return joints, q_candidate, tilt_deg, status

        return None, None, current_tilt_deg, last_err

    def _execute_frontier_prefix_for_inspection(
        self,
        q_path,
        xyz_path,
        reason,
        blocked_xyz=None,
        current_grasp_q=None,
        current_tilt_deg=None,
    ):
        """Execute only the already validated safe prefix and stop.

        V70 diagnostic behavior: when incremental planning reaches a local
        frontier, do not discard all successful virtual work.  Move the real
        robot to the last collision-checked / branch-continuous state, then
        stop before the rejected point so the user can visually inspect the
        limiting geometry or kinematic posture.

        The rejected/colliding point is NEVER executed.  Gripper state is not
        modified.
        """
        if len(q_path) < 2 or len(xyz_path) < 2:
            self.emit(
                "[V70 FRONTIER] No safe prefix beyond the current pose; "
                "nothing will move"
            )
            return False

        last_safe = [float(v) for v in xyz_path[-1]]
        gap_mm = None
        if blocked_xyz is not None:
            try:
                gap_mm = self._xyz_distance(last_safe, blocked_xyz) * 1000.0
            except Exception:
                gap_mm = None

        reason_text = str(reason)
        if "COLLISION/SELF-COLLISION" in reason_text:
            kind = "COLLISION FRONTIER"
        elif "LOCAL BRANCH ENDED" in reason_text or "KINEMATIC/IK" in reason_text:
            kind = "KINEMATIC FRONTIER"
        else:
            kind = "LOCAL PLANNING FRONTIER"

        gap_text = f" | rejected step gap={gap_mm:.1f}mm" if gap_mm is not None else ""
        self.emit(
            f"[V70 FRONTIER LOCATED] {kind} | last-safe TCP="
            f"[{last_safe[0]:+.3f},{last_safe[1]:+.3f},{last_safe[2]:+.3f}]"
            f"{gap_text} | reason={reason_text}"
        )
        self.emit(
            f"[V70 FRONTIER EXECUTE] executing {len(q_path)-1} validated "
            "segment(s) only; rejected point will NOT be executed; gripper preserved"
        )

        traj = self._make_incremental_joint_trajectory(q_path, xyz_path)
        if traj is None:
            self.emit("[V70 FRONTIER] Safe-prefix trajectory is empty; nothing will move")
            return False

        self._execute_trajectory(traj)

        end_xyz, _ = self.current_tcp_pose()
        frontier_err = self._xyz_distance(end_xyz, last_safe)
        self.emit(
            f"[V70 INSPECTION STOP] actual TCP="
            f"[{end_xyz[0]:+.3f},{end_xyz[1]:+.3f},{end_xyz[2]:+.3f}] | "
            f"last-safe error={frontier_err*1000.0:.1f}mm | {kind}"
        )
        self.emit(
            "[V70 INSPECTION] Robot intentionally stopped at the last accepted "
            "state. Inspect the scene/joints; do not open the gripper unless desired."
        )

        # The physical robot is now at this locally accepted grasp orientation.
        if current_grasp_q is not None:
            self.active_grasp_quat = quat_normalize(current_grasp_q)

        return True

    def move_above_bin(self, bin_name, clearance=BIN_RELEASE_CLEARANCE, solver_mode="bio", allow_trac_fallback=True):
        """V70 branch-aware BioIK incremental transfer.

        Priority order:
          1. Continue toward the bin with the *current* local orientation.
          2. If that local branch ends, keep XYZ continuation tiny (2.5 mm)
             and relax only roll/pitch inside a <=10 deg cone: 3, 6, 10 deg.
          3. Rank all valid orientation candidates by minimum joint movement.
          4. Only if orientation relaxation cannot continue the branch, use
             the old bounded +Z / lateral local detours.
          5. TRAC-IK remains a bounded per-waypoint fallback only.

        Normal planning is virtual. If the full route succeeds, the physical arm
        receives one complete trajectory. If a terminal local frontier is found,
        V70 diagnostic mode executes only the already validated safe prefix and
        intentionally stops before the rejected point for visual inspection.
        """
        if bin_name not in BINS:
            raise ValueError(
                f"Unknown bin '{bin_name}'. Valid bins: {list(BINS)}"
            )

        solver_mode = str(solver_mode).strip().lower()
        if solver_mode not in ("bio", "trac"):
            raise ValueError(f"Unknown solver_mode: {solver_mode}")
        solver_label = "BioIK" if solver_mode == "bio" else "TRAC-IK"

        bx, by, rim_z = map(float, BINS[bin_name])
        target_xyz = [bx, by, float(rim_z) + float(clearance)]
        current_xyz, current_tcp_q = self.current_tcp_pose()
        current_xyz = [float(v) for v in current_xyz]

        base_grasp_q = quat_normalize(self.active_grasp_quat or current_tcp_q)
        current_grasp_q = tuple(base_grasp_q)
        current_tilt_deg = (0.0, 0.0)
        max_tilt_used_deg = 0.0

        live_js = self._current_joint_state_copy()
        current_arm = self._arm_positions_from_joint_state(live_js)
        seed_js = self._seed_with_arm_positions(live_js, current_arm)

        xyz_path = [list(current_xyz)]
        q_path = [list(current_arm)]

        base_step = 0.020
        min_step = 0.0025
        max_step = 0.030
        step = base_step
        goal_tol = 0.0015
        max_iters = 140
        z_ceiling = max(0.240, current_xyz[2] + 0.080, target_xyz[2] + 0.060)
        z_ceiling = min(z_ceiling, 0.270)

        self.emit(
            f"[V70 BRANCH-AWARE IK/{solver_label}] {bin_name} | "
            f"start=[{current_xyz[0]:+.3f},{current_xyz[1]:+.3f},{current_xyz[2]:+.3f}] | "
            f"target=[{target_xyz[0]:+.3f},{target_xyz[1]:+.3f},{target_xyz[2]:+.3f}] | "
            f"step=20->10->5->2.5mm | orientation_relax=3/6/10deg | "
            f"local_z_ceiling={z_ceiling:.3f}"
        )

        t0 = time.perf_counter()
        consecutive_detours = 0

        for iteration in range(1, max_iters + 1):
            remaining = self._xyz_distance(current_xyz, target_xyz)
            if remaining <= goal_tol:
                break

            vec = [target_xyz[i] - current_xyz[i] for i in range(3)]
            norm = max(1e-12, math.sqrt(sum(v * v for v in vec)))
            unit = [v / norm for v in vec]

            accepted = None
            failure_text = None
            trial = min(step, remaining)

            trial_sizes = []
            s = trial
            while s >= min_step - 1e-9:
                if not trial_sizes or abs(s - trial_sizes[-1]) > 1e-6:
                    trial_sizes.append(s)
                if s <= min_step + 1e-9:
                    break
                s = max(min_step, s * 0.5)

            # 1) Normal continuation with the current local orientation.
            for ds in trial_sizes:
                candidate = [
                    current_xyz[i] + unit[i] * min(ds, remaining)
                    for i in range(3)
                ]
                if remaining <= ds + 1e-9:
                    candidate = list(target_xyz)

                joints, err = self._try_incremental_ik_point(
                    candidate,
                    current_grasp_q,
                    seed_js,
                    current_arm,
                    timeout_s=0.12,
                    max_joint_step_deg=14.0,
                    solver_mode=solver_mode,
                    allow_trac_fallback=bool(allow_trac_fallback),
                )
                if joints is not None:
                    accepted = (
                        candidate, joints, ds, "direct",
                        current_grasp_q, current_tilt_deg,
                    )
                    break
                failure_text = str(err)

            if accepted is None:
                probe_ds = min(min_step, remaining)
                probe_xyz = [
                    current_xyz[i] + unit[i] * probe_ds
                    for i in range(3)
                ]

                # 2) V64's key change: before moving XYZ out of the way, search
                # a tiny orientation cone at the next local continuation point.
                joints, relaxed_q, relaxed_tilt, relax_status = self._try_orientation_relaxation(
                    probe_xyz,
                    base_grasp_q,
                    current_tilt_deg,
                    seed_js,
                    current_arm,
                    solver_mode=solver_mode,
                    allow_trac_fallback=bool(allow_trac_fallback),
                    max_joint_step_deg=14.0,
                )
                if joints is not None:
                    accepted = (
                        probe_xyz, joints, probe_ds,
                        f"orient({relaxed_tilt[0]:+.1f},{relaxed_tilt[1]:+.1f})",
                        relaxed_q, relaxed_tilt,
                    )
                else:
                    reason = self._diagnose_incremental_failure(
                        probe_xyz, current_grasp_q, seed_js, current_arm,
                        solver_mode=solver_mode,
                        max_joint_step_deg=14.0,
                    )
                    self.emit(
                        f"[V70 LOCAL BLOCK] iter={iteration} | "
                        f"remaining={remaining*1000.0:.0f}mm | {reason} | "
                        f"orientation_relax_failed={relax_status} | last={failure_text}"
                    )

                    # 3) Only now use bounded XYZ detours.  They retain the
                    # current relaxed orientation rather than snapping back.
                    detours = []
                    for dz in (0.010, 0.020):
                        z = min(z_ceiling, current_xyz[2] + dz)
                        if z > current_xyz[2] + 0.002:
                            detours.append((f"up+{dz*1000:.0f}mm", [
                                current_xyz[0], current_xyz[1], z
                            ]))

                    ux, uy = unit[0], unit[1]
                    xy_norm = math.hypot(ux, uy)
                    if xy_norm > 1e-8:
                        px, py = -uy / xy_norm, ux / xy_norm
                        fwd = min(0.005, remaining)
                        for sign in (+1.0, -1.0):
                            detours.append((
                                "lateral+" if sign > 0 else "lateral-",
                                [
                                    current_xyz[0] + ux * fwd + px * 0.010 * sign,
                                    current_xyz[1] + uy * fwd + py * 0.010 * sign,
                                    current_xyz[2],
                                ],
                            ))

                    for detour_name, candidate in detours:
                        joints, err = self._try_incremental_ik_point(
                            candidate,
                            current_grasp_q,
                            seed_js,
                            current_arm,
                            timeout_s=0.15,
                            max_joint_step_deg=12.0,
                            solver_mode=solver_mode,
                            allow_trac_fallback=bool(allow_trac_fallback),
                        )
                        if joints is not None:
                            accepted = (
                                candidate,
                                joints,
                                self._xyz_distance(current_xyz, candidate),
                                detour_name,
                                current_grasp_q,
                                current_tilt_deg,
                            )
                            break
                        failure_text = str(err)

                    if accepted is None:
                        # V70 diagnostic mode: the rejected probe is not executed.
                        # Move only to the last fully validated state so the user
                        # can inspect the collision/kinematic frontier in MuJoCo.
                        moved = self._execute_frontier_prefix_for_inspection(
                            q_path,
                            xyz_path,
                            reason=(
                                f"{reason}; orientation={relax_status}; "
                                f"last={failure_text}"
                            ),
                            blocked_xyz=probe_xyz,
                            current_grasp_q=current_grasp_q,
                            current_tilt_deg=current_tilt_deg,
                        )
                        if moved:
                            return False
                        raise RuntimeError(
                            "V70 branch-aware IK stopped: no local continuation "
                            f"at iter {iteration}; {reason}; orientation={relax_status}; "
                            f"last={failure_text}. No validated prefix exists."
                        )

            candidate, joints, used_step, mode, next_grasp_q, next_tilt_deg = accepted
            max_dq = self._max_joint_delta(current_arm, joints)

            current_xyz = [float(v) for v in candidate]
            current_arm = [float(v) for v in joints]
            current_grasp_q = quat_normalize(next_grasp_q)
            current_tilt_deg = (float(next_tilt_deg[0]), float(next_tilt_deg[1]))
            max_tilt_used_deg = max(max_tilt_used_deg, math.hypot(*current_tilt_deg))
            seed_js = self._seed_with_arm_positions(seed_js, current_arm)
            xyz_path.append(list(current_xyz))
            q_path.append(list(current_arm))

            if mode == "direct":
                consecutive_detours = 0
                step = min(max_step, max(base_step, float(used_step) * 1.5))
            elif mode.startswith("orient("):
                # We advanced toward the goal, so this is not an XYZ detour.
                consecutive_detours = 0
                step = 0.010
            else:
                consecutive_detours += 1
                step = 0.010
                if consecutive_detours > 6:
                    moved = self._execute_frontier_prefix_for_inspection(
                        q_path, xyz_path,
                        reason="LOCAL PLANNING FRONTIER: too many consecutive XYZ detours",
                        blocked_xyz=None,
                        current_grasp_q=current_grasp_q,
                        current_tilt_deg=current_tilt_deg,
                    )
                    if moved:
                        return False
                    raise RuntimeError(
                        "V70 branch-aware IK aborted: too many consecutive XYZ detours."
                    )

            if iteration == 1 or iteration % 5 == 0 or mode != "direct":
                self.emit(
                    f"[V70 IK STEP] #{iteration} | mode={mode} | "
                    f"XYZ=[{current_xyz[0]:+.3f},{current_xyz[1]:+.3f},{current_xyz[2]:+.3f}] | "
                    f"remain={self._xyz_distance(current_xyz,target_xyz)*1000.0:.0f}mm | "
                    f"tilt=({current_tilt_deg[0]:+.1f},{current_tilt_deg[1]:+.1f})deg | "
                    f"max_dq={math.degrees(max_dq):.1f}deg"
                )
        else:
            moved = self._execute_frontier_prefix_for_inspection(
                q_path, xyz_path,
                reason="LOCAL PLANNING FRONTIER: incremental IK iteration limit",
                blocked_xyz=None,
                current_grasp_q=current_grasp_q,
                current_tilt_deg=current_tilt_deg,
            )
            if moved:
                return False
            raise RuntimeError(
                "V70 branch-aware IK exceeded iteration limit with no executable prefix."
            )

        final_error = self._xyz_distance(current_xyz, target_xyz)
        if final_error > goal_tol:
            raise RuntimeError(
                f"V70 branch-aware IK ended {final_error*1000.0:.1f}mm from target. "
                "No physical motion executed."
            )

        max_step_q = 0.0
        total_joint_travel = 0.0
        for a, b in zip(q_path[:-1], q_path[1:]):
            ds = [abs(float(b[i]) - float(a[i])) for i in range(7)]
            max_step_q = max(max_step_q, max(ds))
            total_joint_travel += sum(ds)

        if max_step_q > math.radians(14.5):
            raise RuntimeError(
                f"V70 path rejected before execution: max joint step "
                f"{math.degrees(max_step_q):.1f}deg"
            )

        plan_s = time.perf_counter() - t0
        traj = self._make_incremental_joint_trajectory(q_path, xyz_path)

        self.emit(
            f"[V70 PLAN READY] {len(q_path)-1} IK segment(s) | "
            f"planning={plan_s:.2f}s | max_joint_step={math.degrees(max_step_q):.1f}deg | "
            f"max_tilt={max_tilt_used_deg:.1f}deg | "
            f"joint_travel={math.degrees(total_joint_travel):.0f}deg"
        )
        self.emit(
            "[V70 EXECUTE ONCE] branch-aware path validated; "
            "sending one FollowJointTrajectory goal"
        )
        self._execute_trajectory(traj)

        deadline = time.time() + 3.0
        end_xyz = None
        dist = float("inf")
        while time.time() < deadline:
            end_xyz, _ = self.current_tcp_pose()
            dist = self._xyz_distance(end_xyz, target_xyz)
            if dist <= 0.035:
                break
            time.sleep(0.05)

        self.emit(
            f"[V70 BIN VERIFY] {bin_name} | TCP="
            f"[{end_xyz[0]:+.3f},{end_xyz[1]:+.3f},{end_xyz[2]:+.3f}] | "
            f"target error={dist*1000.0:.1f}mm | "
            f"final_tilt=({current_tilt_deg[0]:+.1f},{current_tilt_deg[1]:+.1f})deg"
        )
        if dist > 0.050:
            raise RuntimeError(
                f"V70 bin transit verification failed: TCP is "
                f"{dist*1000.0:.1f}mm from target"
            )

        # Preserve the actual local grasp orientation reached by V70 so any
        # subsequent arm skill continues from the same branch/orientation.
        self.active_grasp_quat = quat_normalize(current_grasp_q)
        self.emit(
            f"move_above_bin complete | {bin_name} | {solver_label} branch-aware IK | "
            f"tilt=({current_tilt_deg[0]:+.1f},{current_tilt_deg[1]:+.1f})deg"
        )
        return True

    def _compute_ik_with_client(
        self,
        client,
        service_label,
        xyz,
        quat_xyzw,
        timeout_s=IK_TIMEOUT_S,
        avoid_collisions=True,
        seed_joint_state=None,
    ):
        if not client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(f"{service_label} unavailable")

        req = GetPositionIK.Request()
        ik = req.ik_request
        ik.group_name = "arm"
        ik.ik_link_name = "tcp_link"
        ik.avoid_collisions = bool(avoid_collisions)

        js = seed_joint_state if seed_joint_state is not None else self._current_joint_state_copy()
        ik.robot_state.joint_state = js

        ik.pose_stamped.header.frame_id = "base_link"
        ik.pose_stamped.header.stamp = self.node.get_clock().now().to_msg()
        ik.pose_stamped.pose.position.x = float(xyz[0])
        ik.pose_stamped.pose.position.y = float(xyz[1])
        ik.pose_stamped.pose.position.z = float(xyz[2])

        q = quat_normalize(quat_xyzw)
        ik.pose_stamped.pose.orientation.x = float(q[0])
        ik.pose_stamped.pose.orientation.y = float(q[1])
        ik.pose_stamped.pose.orientation.z = float(q[2])
        ik.pose_stamped.pose.orientation.w = float(q[3])

        sec = int(timeout_s)
        nsec = int((float(timeout_s) - sec) * 1_000_000_000)
        ik.timeout.sec = sec
        ik.timeout.nanosec = nsec

        response = self.wait_future(
            client.call_async(req),
            float(timeout_s) + 2.0,
            service_label,
        )
        if response is None:
            raise RuntimeError(f"{service_label} returned no response")
        if int(response.error_code.val) != 1:
            return None, moveit_error_text(response.error_code)

        name_to_pos = {
            str(n): float(p)
            for n, p in zip(
                response.solution.joint_state.name,
                response.solution.joint_state.position,
            )
        }
        if not all(name in name_to_pos for name in ARM_JOINT_NAMES):
            return None, "IK returned incomplete arm state: " + ",".join(name_to_pos.keys())
        return [name_to_pos[name] for name in ARM_JOINT_NAMES], "SUCCESS"

    def _compute_ik(
        self,
        xyz,
        quat_xyzw,
        timeout_s=IK_TIMEOUT_S,
        avoid_collisions=True,
        seed_joint_state=None,
    ):
        """Primary BioIK service at root /compute_ik."""
        return self._compute_ik_with_client(
            self.ik_client,
            "/compute_ik (BioIK)",
            xyz,
            quat_xyzw,
            timeout_s,
            avoid_collisions,
            seed_joint_state,
        )

    def _compute_trac_ik(
        self,
        xyz,
        quat_xyzw,
        timeout_s=IK_TIMEOUT_S,
        avoid_collisions=True,
        seed_joint_state=None,
    ):
        """Reference/fallback TRAC-IK service in /trac_ref."""
        return self._compute_ik_with_client(
            self.trac_ik_client,
            "/trac_ref/compute_ik (TRAC-IK)",
            xyz,
            quat_xyzw,
            timeout_s,
            avoid_collisions,
            seed_joint_state,
        )


    # ========================================================
    # V63 BIOIK LAB (PRIMARY)
    # ========================================================

    def _bio_make_seed(self, mode="current"):
        # Same seed semantics as the TRAC lab for fair A/B comparison.
        return self._trac_make_seed(mode)

    def _ik_report_solution(self, prefix, label, solution, seed_js, elapsed_s):
        seed = self._trac_arm_positions_from_js(seed_js)
        delta = [float(a) - float(b) for a, b in zip(solution, seed)]
        l2 = math.sqrt(sum(d * d for d in delta))
        max_abs = max(abs(d) for d in delta)
        self.emit(
            f"[{prefix} IK PASS] {label} | {elapsed_s*1000.0:.1f} ms | "
            f"seed-distance L2={l2:.3f} rad | max ΔJ={max_abs:.3f} rad"
        )
        self.emit(
            f"[{prefix} SOLUTION] [" + ", ".join(f"{float(v):+.3f}" for v in solution) + "]"
        )
        return l2, max_abs

    def bio_self_test(self, seed_mode="current", avoid_collisions=True, timeout_s=0.05):
        xyz, q = self.current_tcp_pose()
        seed_js = self._bio_make_seed(seed_mode)
        self.emit(
            f"[BIO SELF-TEST] current TCP | seed={seed_mode} | "
            f"collision={bool(avoid_collisions)} | timeout={float(timeout_s):.3f}s"
        )
        t0 = time.perf_counter()
        joints, err = self._compute_ik(
            xyz, q, timeout_s=float(timeout_s),
            avoid_collisions=bool(avoid_collisions), seed_joint_state=seed_js,
        )
        dt = time.perf_counter() - t0
        if joints is None:
            self.emit(f"[BIO IK FAIL] CURRENT TCP | {dt*1000.0:.1f} ms | {err}")
            return False
        self._ik_report_solution("BIO", "CURRENT TCP", joints, seed_js, dt)
        return True

    def bio_object_probe(
        self, object_name, height=0.12, yaw_deg=0.0, seed_mode="current",
        avoid_collisions=True, timeout_s=0.05,
    ):
        obj_xyz, _ = self.object_pose(object_name)
        xyz = [float(obj_xyz[0]), float(obj_xyz[1]), float(obj_xyz[2]) + float(height)]
        q = top_down_quat(math.radians(float(yaw_deg)))
        seed_js = self._bio_make_seed(seed_mode)
        self.emit(
            f"[BIO PROBE] {object_name} | XYZ=[{xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}] | "
            f"yaw={float(yaw_deg):+.0f}deg | seed={seed_mode} | "
            f"collision={bool(avoid_collisions)} | timeout={float(timeout_s):.3f}s"
        )
        t0 = time.perf_counter()
        joints, err = self._compute_ik(
            xyz, q, timeout_s=float(timeout_s),
            avoid_collisions=bool(avoid_collisions), seed_joint_state=seed_js,
        )
        dt = time.perf_counter() - t0
        if joints is None:
            self.emit(f"[BIO IK FAIL] {object_name} | {dt*1000.0:.1f} ms | {err}")
            return False
        self._ik_report_solution("BIO", object_name, joints, seed_js, dt)
        return True

    def bio_benchmark(
        self, object_name, height=0.12, yaw_deg=0.0, seed_mode="current",
        avoid_collisions=True, timeout_s=0.05, runs=20,
    ):
        runs = max(1, min(100, int(runs)))
        obj_xyz, _ = self.object_pose(object_name)
        xyz = [float(obj_xyz[0]), float(obj_xyz[1]), float(obj_xyz[2]) + float(height)]
        q = top_down_quat(math.radians(float(yaw_deg)))
        times, distances, errors = [], [], {}
        successes = 0
        self.emit(
            f"[BIO BENCH] {object_name} | runs={runs} | seed={seed_mode} | "
            f"collision={bool(avoid_collisions)} | timeout={float(timeout_s):.3f}s"
        )
        for _ in range(runs):
            seed_js = self._bio_make_seed(seed_mode)
            seed = self._trac_arm_positions_from_js(seed_js)
            t0 = time.perf_counter()
            joints, err = self._compute_ik(
                xyz, q, timeout_s=float(timeout_s),
                avoid_collisions=bool(avoid_collisions), seed_joint_state=seed_js,
            )
            dt = time.perf_counter() - t0
            times.append(dt)
            if joints is None:
                errors[err] = errors.get(err, 0) + 1
                continue
            successes += 1
            delta = [float(a) - float(b) for a, b in zip(joints, seed)]
            distances.append(math.sqrt(sum(d*d for d in delta)))
        avg_ms = (sum(times) / len(times) * 1000.0) if times else float("nan")
        self.emit(
            f"[BIO BENCH RESULT] success={successes}/{runs} | avg={avg_ms:.1f} ms"
        )
        if distances:
            self.emit(
                f"[BIO BENCH RESULT] seed-distance avg={sum(distances)/len(distances):.3f} rad | "
                f"min={min(distances):.3f} | max={max(distances):.3f}"
            )
        if errors:
            self.emit("[BIO BENCH FAILURES] " + " | ".join(f"{k}: {v}" for k,v in sorted(errors.items())))
        return successes == runs

    def bio_move_above(self, object_name, height=0.12):
        self.emit("[BIO SKILL] Move Above using primary BioIK-configured MoveIt")
        return self.move_above(object_name, float(height))

    def bio_approach(self, object_name):
        self.emit("[BIO SKILL] collision-checked Cartesian Approach")
        return self.approach(object_name, None)

    def bio_lift(self, dz=0.10):
        self.emit("[BIO SKILL] proven strict-Cartesian Lift")
        return self.lift(float(dz))

    def bio_move_above_bin(self, bin_name):
        self.emit("[BIO SKILL] BioIK-first incremental bin transfer; TRAC fallback enabled")
        return self.move_above_bin(
            bin_name, solver_mode="bio", allow_trac_fallback=True
        )

    # ========================================================
    # V52 TRAC-IK LAB
    # ========================================================

    def bio_readiness(self):
        try:
            ik_ready = bool(self.ik_client.service_is_ready())
        except Exception:
            ik_ready = False
        solver = os.environ.get("NERO_BIO_IK_SOLVER", "BioIK")
        mode = os.environ.get("NERO_BIO_IK_MODE", "unknown")
        timeout = os.environ.get("NERO_BIO_IK_TIMEOUT", "unknown")
        weights = os.environ.get("NERO_BIO_IK_WEIGHTS", "unknown")
        return ik_ready, solver, mode, timeout, weights

    def trac_readiness(self):
        try:
            ik_ready = bool(self.trac_ik_client.service_is_ready())
        except Exception:
            ik_ready = False
        solver = os.environ.get("NERO_TRAC_IK_SOLVER", "TRAC-IK")
        solve_type = os.environ.get("NERO_TRAC_IK_SOLVE_TYPE", "unknown")
        timeout = os.environ.get("NERO_TRAC_IK_TIMEOUT", "unknown")
        return ik_ready, solver, solve_type, timeout


    def _trac_make_seed(self, mode="current"):
        """
        Return a complete JointState seed while changing only J1..J7.
        Supported modes: current, zero, bent.
        """
        mode = str(mode).strip().lower()
        live = self._current_joint_state_copy()

        if mode == "current":
            return live

        seed = JointState()
        seed.header = live.header
        seed.header.stamp = self.node.get_clock().now().to_msg()
        seed.name = list(live.name)
        seed.position = list(live.position)
        seed.velocity = list(live.velocity)
        seed.effort = list(live.effort)

        idx = {str(n): i for i, n in enumerate(seed.name)}

        if mode == "zero":
            arm = [0.0] * 7
        elif mode in ("bent", "bent-elbow", "bent_elbow"):
            arm = list(TRAC_BENT_SEED)
        else:
            raise RuntimeError(
                f"Unknown TRAC seed mode: {mode}"
            )

        for name, value in zip(ARM_JOINT_NAMES, arm):
            if name in idx:
                seed.position[idx[name]] = float(value)
            else:
                seed.name.append(name)
                seed.position.append(float(value))

        return seed

    @staticmethod
    def _trac_arm_positions_from_js(js):
        mp = {
            str(n): float(p)
            for n, p in zip(js.name, js.position)
        }
        if not all(name in mp for name in ARM_JOINT_NAMES):
            raise RuntimeError("JointState seed missing arm joints")
        return [mp[name] for name in ARM_JOINT_NAMES]

    def _trac_report_solution(
        self,
        label,
        solution,
        seed_js,
        elapsed_s,
    ):
        seed = self._trac_arm_positions_from_js(seed_js)
        delta = [
            float(a) - float(b)
            for a, b in zip(solution, seed)
        ]
        l2 = math.sqrt(sum(d * d for d in delta))
        max_abs = max(abs(d) for d in delta)

        self.emit(
            f"[TRAC IK PASS] {label} | "
            f"{elapsed_s*1000.0:.1f} ms | "
            f"seed-distance L2={l2:.3f} rad | "
            f"max ΔJ={max_abs:.3f} rad"
        )
        self.emit(
            "[TRAC SOLUTION] ["
            + ", ".join(f"{float(v):+.3f}" for v in solution)
            + "]"
        )
        return l2, max_abs

    def trac_self_test(
        self,
        seed_mode="current",
        avoid_collisions=True,
        timeout_s=TRAC_PANEL_DEFAULT_TIMEOUT_S,
    ):
        """
        Solve IK for the robot's CURRENT tcp_link pose.

        If this fails, the issue is solver/model/plugin configuration rather
        than whether a tabletop target is physically reachable.
        """
        xyz, q = self.current_tcp_pose()
        seed_js = self._trac_make_seed(seed_mode)

        self.emit(
            f"[TRAC SELF-TEST] current tcp pose | "
            f"seed={seed_mode} | collision={bool(avoid_collisions)} | "
            f"timeout={float(timeout_s):.3f}s"
        )

        t0 = time.perf_counter()
        joints, err = self._compute_trac_ik(
            xyz,
            q,
            timeout_s=float(timeout_s),
            avoid_collisions=bool(avoid_collisions),
            seed_joint_state=seed_js,
        )
        dt = time.perf_counter() - t0

        if joints is None:
            self.emit(
                f"[TRAC IK FAIL] CURRENT TCP | "
                f"{dt*1000.0:.1f} ms | {err}"
            )
            return False

        self._trac_report_solution(
            "CURRENT TCP",
            joints,
            seed_js,
            dt,
        )
        return True

    def trac_object_probe(
        self,
        object_name,
        height=0.12,
        yaw_deg=0.0,
        seed_mode="current",
        avoid_collisions=True,
        timeout_s=TRAC_PANEL_DEFAULT_TIMEOUT_S,
    ):
        obj_xyz, _ = self.object_pose(object_name)

        xyz = [
            float(obj_xyz[0]),
            float(obj_xyz[1]),
            float(obj_xyz[2]) + float(height),
        ]
        q = top_down_quat(math.radians(float(yaw_deg)))
        seed_js = self._trac_make_seed(seed_mode)

        self.emit(
            f"[TRAC PROBE] {object_name} | "
            f"XYZ=[{xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}] | "
            f"yaw={float(yaw_deg):+.0f}deg | "
            f"seed={seed_mode} | collision={bool(avoid_collisions)} | "
            f"timeout={float(timeout_s):.3f}s"
        )

        t0 = time.perf_counter()
        joints, err = self._compute_trac_ik(
            xyz,
            q,
            timeout_s=float(timeout_s),
            avoid_collisions=bool(avoid_collisions),
            seed_joint_state=seed_js,
        )
        dt = time.perf_counter() - t0

        if joints is None:
            self.emit(
                f"[TRAC IK FAIL] {object_name} | "
                f"{dt*1000.0:.1f} ms | {err}"
            )
            return False

        self._trac_report_solution(
            object_name,
            joints,
            seed_js,
            dt,
        )
        return True

    def trac_benchmark(
        self,
        object_name,
        height=0.12,
        yaw_deg=0.0,
        seed_mode="current",
        avoid_collisions=True,
        timeout_s=TRAC_PANEL_DEFAULT_TIMEOUT_S,
        runs=20,
    ):
        runs = max(1, min(100, int(runs)))
        obj_xyz, _ = self.object_pose(object_name)

        xyz = [
            float(obj_xyz[0]),
            float(obj_xyz[1]),
            float(obj_xyz[2]) + float(height),
        ]
        q = top_down_quat(math.radians(float(yaw_deg)))

        times = []
        distances = []
        successes = 0
        errors = {}

        self.emit(
            f"[TRAC BENCH] {object_name} | runs={runs} | "
            f"seed={seed_mode} | collision={bool(avoid_collisions)} | "
            f"timeout={float(timeout_s):.3f}s"
        )

        for i in range(runs):
            seed_js = self._trac_make_seed(seed_mode)
            seed = self._trac_arm_positions_from_js(seed_js)

            t0 = time.perf_counter()
            joints, err = self._compute_trac_ik(
                xyz,
                q,
                timeout_s=float(timeout_s),
                avoid_collisions=bool(avoid_collisions),
                seed_joint_state=seed_js,
            )
            dt = time.perf_counter() - t0
            times.append(dt)

            if joints is None:
                errors[err] = errors.get(err, 0) + 1
                continue

            successes += 1
            delta = [
                float(a) - float(b)
                for a, b in zip(joints, seed)
            ]
            distances.append(
                math.sqrt(sum(d * d for d in delta))
            )

        avg_ms = (
            sum(times) / len(times) * 1000.0
            if times
            else float("nan")
        )

        self.emit(
            f"[TRAC BENCH RESULT] success={successes}/{runs} "
            f"({100.0*successes/runs:.1f}%) | "
            f"avg call={avg_ms:.1f} ms"
        )

        if distances:
            self.emit(
                f"[TRAC BENCH RESULT] seed-distance "
                f"avg={sum(distances)/len(distances):.3f} rad | "
                f"min={min(distances):.3f} | "
                f"max={max(distances):.3f}"
            )

        if errors:
            self.emit(
                "[TRAC BENCH FAILURES] "
                + " | ".join(
                    f"{k}: {v}"
                    for k, v in sorted(errors.items())
                )
            )

        return successes == runs

    def trac_move_above(
        self,
        object_name,
        height=0.12,
    ):
        obj_xyz, _ = self.object_pose(object_name)
        xyz = [float(obj_xyz[0]), float(obj_xyz[1]), float(obj_xyz[2]) + float(height)]
        q = top_down_quat(0.0)
        seed_js = self._current_joint_state_copy()
        self.emit(f"[TRAC SKILL] target IK via /trac_ref/compute_ik -> {object_name}")
        joints, err = self._compute_trac_ik(
            xyz, q, timeout_s=0.15, avoid_collisions=True, seed_joint_state=seed_js
        )
        if joints is None:
            raise RuntimeError("TRAC Move Above IK failed: " + str(err))
        traj = self._plan_joint_solution(joints, velocity_scale=0.16, accel_scale=0.16)
        self._execute_trajectory(traj)
        self.active_grasp_quat = quat_normalize(q)
        self.emit("TRAC move_above complete")
        return True

    def trac_approach(self, object_name):
        self.emit(
            "[TRAC TAB] Approach uses the common collision-checked Cartesian primitive; "
            "solver-specific comparison is provided by TRAC Probe/Benchmark/Move Above."
        )
        return self.approach(object_name, None)

    def trac_lift(self, dz=0.10):
        self.emit("[TRAC TAB] Lift uses the common proven strict-Cartesian lift.")
        return self.lift(float(dz))

    def trac_move_above_bin(self, bin_name):
        self.emit("[TRAC TAB] TRAC-only incremental bin transfer (no BioIK fallback)")
        return self.move_above_bin(
            bin_name, solver_mode="trac", allow_trac_fallback=False
        )

    def _build_joint_goal(
        self,
        positions,
        velocity_scale=0.22,
        accel_scale=0.22,
        planner_id="",
        planning_attempts=6,
        planning_time=6.0,
    ):
        goal = MoveGroup.Goal()
        req = goal.request

        req.group_name = "arm"
        req.pipeline_id = "ompl"
        req.num_planning_attempts = 10
        req.allowed_planning_time = 6.0
        req.max_velocity_scaling_factor = float(velocity_scale)
        req.max_acceleration_scaling_factor = float(accel_scale)
        req.start_state.is_diff = True

        req.workspace_parameters.header.frame_id = "base_link"
        req.workspace_parameters.min_corner.x = -1.5
        req.workspace_parameters.min_corner.y = -1.5
        req.workspace_parameters.min_corner.z = -0.2
        req.workspace_parameters.max_corner.x = 1.5
        req.workspace_parameters.max_corner.y = 1.5
        req.workspace_parameters.max_corner.z = 1.8

        c = Constraints()

        for name, pos in zip(ARM_JOINT_NAMES, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = 0.008
            jc.tolerance_below = 0.008
            jc.weight = 1.0
            c.joint_constraints.append(jc)

        req.goal_constraints = [c]

        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        return goal

    def _plan_joint_solution(
        self,
        positions,
        velocity_scale=0.22,
        accel_scale=0.22,
        planner_id="",
        planning_attempts=6,
        planning_time=6.0,
    ):
        if not self.move_client.wait_for_server(timeout_sec=3.0):
            raise RuntimeError("/move_action unavailable")

        goal = self._build_joint_goal(
            positions,
            velocity_scale=velocity_scale,
            accel_scale=accel_scale,
            planner_id=planner_id,
            planning_attempts=planning_attempts,
            planning_time=planning_time,
        )

        pname = planner_id if planner_id else "MoveIt/OMPL default"
        self.emit(
            f"OMPL planning to prevalidated IK joint solution | planner={pname}..."
        )

        gh = self.wait_future(
            self.move_client.send_goal_async(goal),
            8.0,
            "MoveGroup joint-goal response",
        )

        if gh is None or not gh.accepted:
            raise RuntimeError(
                "MoveGroup rejected IK joint-goal planning request"
            )

        wrapped = self.wait_future(
            gh.get_result_async(),
            20.0,
            "MoveGroup joint-goal planning",
        )

        if wrapped is None:
            raise RuntimeError(
                "MoveGroup returned no joint-goal planning result"
            )

        result = wrapped.result

        if int(result.error_code.val) != 1:
            raise RuntimeError(
                "JOINT-GOAL PLANNING FAILED: "
                + moveit_error_text(result.error_code)
            )

        traj = result.planned_trajectory
        n = len(traj.joint_trajectory.points)

        if n == 0:
            self.emit("Joint IK goal already satisfied")
            return None

        self.emit(
            f"Joint-space plan OK: {n} points | "
            f"{float(result.planning_time):.3f}s"
        )

        return traj

    def _search_grasp_ik(
        self,
        object_name,
        requested_height=MOVE_ABOVE_HEIGHT,
    ):
        """V70: rank all exact-top-down IK candidates before accepting one."""
        obj_xyz,_=self.object_pose(object_name)
        heights=[]
        for h in (requested_height,)+IK_HEIGHT_CANDIDATES:
            h=float(h)
            if all(abs(h-existing)>1e-6 for existing in heights): heights.append(h)

        def collect_for_orientation(tilt_deg,tilt_spec=None):
            pool=[]
            for h in heights:
                xyz=[obj_xyz[0],obj_xyz[1],obj_xyz[2]+h]
                for yaw in IK_YAW_CANDIDATES:
                    if tilt_spec is None:
                        q=top_down_quat(yaw); label='TOP-DOWN'
                    else:
                        axis,angle,label=tilt_spec
                        q=top_down_quat(yaw,tilt_axis=axis,tilt_rad=angle)
                    cands,errs,current=self._v70_ranked_pose_ik_candidates(
                        xyz,q,max_candidates=2,timeout_s=IK_TIMEOUT_S,
                    )
                    for score,joints,seed_tag,metrics in cands:
                        # Keep natural motion/reserve dominant; use requested-height
                        # proximity and smaller absolute yaw as deterministic tie-breaks.
                        full_score=tuple(score)+(
                            abs(float(h)-float(requested_height)),
                            abs(math.atan2(math.sin(yaw),math.cos(yaw))),
                        )
                        pool.append((full_score,joints,q,xyz,float(tilt_deg),h,yaw,seed_tag,metrics,label))
            pool.sort(key=lambda item:item[0])
            return pool

        exact=collect_for_orientation(0.0,None)
        if exact:
            score,joints,q,xyz,tilt_deg,h,yaw,seed_tag,m,label=exact[0]
            self.emit(
                f"[V70 IK FOUND] exact TOP-DOWN | seed={seed_tag} | "
                f"height={h*1000:.0f}mm | yaw={math.degrees(yaw):+.0f}deg | "
                f"max_dJ={max(m['delta_deg']):.1f}deg | wrist_reserve={m['min_wrist_reserve_deg']:.1f}deg"
            )
            return joints,q,xyz,0.0

        tilt=math.radians(IK_SMALL_TILT_DEG)
        specs=(
            ((1.0,0.0,0.0),+tilt,'+X'),((1.0,0.0,0.0),-tilt,'-X'),
            ((0.0,1.0,0.0),+tilt,'+Y'),((0.0,1.0,0.0),-tilt,'-Y'),
        )
        near=[]
        for spec in specs: near.extend(collect_for_orientation(IK_SMALL_TILT_DEG,spec))
        near.sort(key=lambda item:item[0])
        if near:
            score,joints,q,xyz,tilt_deg,h,yaw,seed_tag,m,label=near[0]
            self.emit(
                f"[V70 IK FOUND] 15deg near-top-down {label} | seed={seed_tag} | "
                f"height={h*1000:.0f}mm | yaw={math.degrees(yaw):+.0f}deg | "
                f"max_dJ={max(m['delta_deg']):.1f}deg | wrist_reserve={m['min_wrist_reserve_deg']:.1f}deg"
            )
            return joints,q,xyz,IK_SMALL_TILT_DEG

        raise RuntimeError(
            "No collision-free IK found for exact top-down or <=15deg near-top-down "
            "grasp orientation. MoveIt did not execute any motion."
        )

    def _plan_top_down_above(self, object_name, height=MOVE_ABOVE_HEIGHT):
        joints, q, target, tilt_deg = self._search_grasp_ik(
            object_name,
            requested_height=height,
        )

        self.emit(
            "IK preflight passed; planning collision-free "
            "joint-space path..."
        )

        traj = self._plan_joint_solution(
            joints,
            velocity_scale=0.22,
            accel_scale=0.22,
        )

        return traj, q, target, tilt_deg

    def _cartesian_waypoint_plan(
        self,
        waypoints_xyz,
        target_quat,
        label="Cartesian transit",
        max_step=0.006,
        max_speed=0.07,
    ):
        if not self.cartesian_client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError("/compute_cartesian_path unavailable")

        req = GetCartesianPath.Request()
        req.header.frame_id = "base_link"
        req.group_name = "arm"
        req.link_name = "tcp_link"
        req.start_state.is_diff = True

        q = quat_normalize(target_quat)
        poses = []

        for xyz in waypoints_xyz:
            pose = Pose()
            pose.position.x = float(xyz[0])
            pose.position.y = float(xyz[1])
            pose.position.z = float(xyz[2])
            pose.orientation.x = float(q[0])
            pose.orientation.y = float(q[1])
            pose.orientation.z = float(q[2])
            pose.orientation.w = float(q[3])
            poses.append(pose)

        req.waypoints = poses
        req.max_step = float(max_step)
        req.jump_threshold = 0.0
        req.avoid_collisions = True

        if hasattr(req, "prismatic_jump_threshold"):
            req.prismatic_jump_threshold = 0.0
        if hasattr(req, "revolute_jump_threshold"):
            req.revolute_jump_threshold = 0.0
        if hasattr(req, "max_velocity_scaling_factor"):
            req.max_velocity_scaling_factor = 0.18
        if hasattr(req, "max_acceleration_scaling_factor"):
            req.max_acceleration_scaling_factor = 0.18
        if hasattr(req, "cartesian_speed_limited_link"):
            req.cartesian_speed_limited_link = "tcp_link"
        if hasattr(req, "max_cartesian_speed"):
            req.max_cartesian_speed = float(max_speed)

        self.emit(
            f"[CONTINUOUS] {label}: "
            f"{len(poses)} waypoint(s), fixed grasp orientation"
        )

        response = self.wait_future(
            self.cartesian_client.call_async(req),
            20.0,
            "GetCartesianPath continuous move_above",
        )

        if response is None:
            raise RuntimeError(
                "GetCartesianPath returned no response"
            )

        if int(response.error_code.val) != 1:
            raise RuntimeError(
                "Cartesian service failed: "
                + moveit_error_text(response.error_code)
            )

        fraction = float(response.fraction)

        if fraction < 0.98:
            raise RuntimeError(
                f"Cartesian corridor only "
                f"{fraction*100.0:.1f}% collision-free"
            )

        traj = response.solution
        n = len(traj.joint_trajectory.points)

        if n == 0:
            self.emit(
                "[CONTINUOUS] target already satisfied"
            )
            return None

        self.emit(
            f"[CONTINUOUS OK] {fraction*100.0:.1f}% | "
            f"{n} trajectory points"
        )

        return traj

    def _try_continuous_move_above(
        self,
        object_name,
        height=MOVE_ABOVE_HEIGHT,
    ):
        if self.active_grasp_quat is None:
            return None, None

        current_xyz, _ = self.current_tcp_pose()
        obj_xyz, _ = self.object_pose(object_name)

        target = [
            float(obj_xyz[0]),
            float(obj_xyz[1]),
            float(obj_xyz[2]) + float(height),
        ]

        safe_z_candidates = []

        for z in (
            max(
                float(current_xyz[2]),
                float(target[2]) + 0.06,
                0.42,
            ),
            max(
                float(current_xyz[2]),
                float(target[2]) + 0.10,
                0.48,
            ),
        ):
            if all(
                abs(z - old) > 1e-6
                for old in safe_z_candidates
            ):
                safe_z_candidates.append(z)

        last_exc = None

        for safe_z in safe_z_candidates:
            waypoints = [
                [
                    float(current_xyz[0]),
                    float(current_xyz[1]),
                    float(safe_z),
                ],
                [
                    float(target[0]),
                    float(target[1]),
                    float(safe_z),
                ],
                [
                    float(target[0]),
                    float(target[1]),
                    float(target[2]),
                ],
            ]

            try:
                traj = self._cartesian_waypoint_plan(
                    waypoints,
                    self.active_grasp_quat,
                    label=(
                        "adjacent-object corridor "
                        f"safe_z={safe_z:.3f}"
                    ),
                    max_step=0.006,
                    max_speed=0.07,
                )
                return traj, target

            except Exception as exc:
                last_exc = exc
                self.emit(
                    "[CONTINUOUS RETRY] "
                    f"safe_z={safe_z:.3f} failed: {exc}"
                )

        if last_exc is not None:
            raise last_exc

        raise RuntimeError(
            "No continuous Cartesian corridor candidate"
        )

    def move_above(self, object_name, height=MOVE_ABOVE_HEIGHT):
        # Preferred path after the first successful grasp pose: preserve
        # orientation/IK branch and move continuously to a nearby object.
        if self.active_grasp_quat is not None:
            try:
                traj, target = self._try_continuous_move_above(
                    object_name, height
                )
                if traj is not None:
                    self._execute_trajectory(traj)
                    self.emit(
                        "move_above complete | CONTINUOUS GRASP ORIENTATION | "
                        f"[{target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}]"
                    )
                    return True
            except Exception as exc:
                self.emit(f"[CONTINUOUS FALLBACK] {exc}")
                self.emit("Falling back to V44 global IK search...")

        traj, q, target, tilt_deg = self._plan_top_down_above(
            object_name, height
        )
        self._execute_trajectory(traj)
        self.active_grasp_quat = q

        mode = (
            "TOP-DOWN"
            if abs(float(tilt_deg)) < 1e-6
            else f"NEAR-TOP-DOWN tilt={tilt_deg:.0f}deg"
        )
        self.emit(
            f"move_above complete | {mode} | "
            f"[{target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}]"
        )
        return True

    def _cartesian_approach_plan(self, target_xyz, target_quat):
        if not self.cartesian_client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError("/compute_cartesian_path unavailable")

        req = GetCartesianPath.Request()
        req.header.frame_id = "base_link"
        req.group_name = "arm"
        req.link_name = "tcp_link"
        req.start_state.is_diff = True

        pose = Pose()
        pose.position.x = float(target_xyz[0])
        pose.position.y = float(target_xyz[1])
        pose.position.z = float(target_xyz[2])
        q = quat_normalize(target_quat)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
        req.waypoints = [pose]
        req.max_step = float(CARTESIAN_MAX_STEP)
        req.jump_threshold = 0.0
        if hasattr(req, "prismatic_jump_threshold"):
            req.prismatic_jump_threshold = 0.0
        if hasattr(req, "revolute_jump_threshold"):
            req.revolute_jump_threshold = 0.0
        req.avoid_collisions = True
        if hasattr(req, "max_velocity_scaling_factor"):
            req.max_velocity_scaling_factor = 0.12
        if hasattr(req, "max_acceleration_scaling_factor"):
            req.max_acceleration_scaling_factor = 0.12
        if hasattr(req, "cartesian_speed_limited_link"):
            req.cartesian_speed_limited_link = "tcp_link"
        if hasattr(req, "max_cartesian_speed"):
            req.max_cartesian_speed = float(CARTESIAN_SPEED)

        self.emit("Computing collision-checked straight Cartesian approach...")
        response = self.wait_future(
            self.cartesian_client.call_async(req),
            12.0,
            "GetCartesianPath",
        )
        if response is None:
            raise RuntimeError("GetCartesianPath returned no response")
        if int(response.error_code.val) != 1:
            raise RuntimeError("CARTESIAN FAILED: " + moveit_error_text(response.error_code))
        fraction = float(response.fraction)
        if fraction < 0.98:
            raise RuntimeError(
                f"CARTESIAN APPROACH BLOCKED: only {fraction*100.0:.1f}% collision-free"
            )
        n = len(response.solution.joint_trajectory.points)
        if n == 0:
            raise RuntimeError("Cartesian approach returned empty trajectory")
        self.emit(f"Cartesian plan OK: {fraction*100.0:.1f}% | {n} points")
        return response.solution

    def approach(self, object_name, z_offset=None):
        obj_xyz, _ = self.object_pose(object_name)
        if z_offset is None:
            z_offset = GRASP_Z_OFFSET.get(object_name, DEFAULT_GRASP_Z_OFFSET)
        target = [obj_xyz[0], obj_xyz[1], obj_xyz[2] + float(z_offset)]
        q = self.active_grasp_quat or top_down_quat(0.0)
        traj = self._cartesian_approach_plan(target, q)
        self._execute_trajectory(traj)
        self.emit(
            "approach complete | straight descent | "
            f"[{target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}]"
        )
        return True
