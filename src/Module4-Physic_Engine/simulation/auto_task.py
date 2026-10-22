"""Module 4 simulation - Autonomous simulation task: actor -> target ROI object -> collision -> analysis.

Simulation = a virtual system with the same physical properties as the real scene, used to
analyse and predict what an interaction would do before doing it for real (where it would be
expensive, slow or dangerous). Instead of dragging objects by hand in the viewer:

  1. Actor: an ROI object double-clicked in the viewer (or --actor=ROI_x). It is carried by
     physical forces (PD controller + gravity compensation through xfrc_applied, force capped
     at MAX_PUSH_FORCE), so its mass and collision hulls take part in the impact.
     Without an actor (headless / --auto) the actor is the agent: a kinematic box, stand-in
     for a robot end-effector.
  2. Target: the ROI object double-clicked next (or --target=ROI_y); by default the ROI object
     closest to the actor (distance to the ROI AABB). A static target (wall, sofa, ...) is
     allowed for an object actor: the actor is thrown against a fixed structure.
  3. The actor moves along a straight line from where it is to the target. The aim height
     inside the target box is the first one a MuJoCo ray from the actor actually hits the
     target at (a table is hit at its top, not between its legs).
  4. It moves into the target until contact, keeps pushing PUSH_DEPTH, holds, then the agent
     retreats / the object actor is released.
  5. It watches the scene until everything is at rest and reports the predicted outcome:
     contact time / point / force / impulse, target displacement and tilt
     (PUSHED / TOPPLED / BLOCKED / IMPACT / MISSED), what happened to the actor, other objects
     set in motion and the Topology Guard faults.

Phases:  [WAIT ->] SETTLE -> APPROACH -> PUSH -> HOLD -> RETREAT -> OBSERVE -> DONE
         WAIT (viewer): double-click the actor, then the target. An object actor skips
         RETREAT (it is released instead).

Usage (from the repository root):
    python scripts/06_run_simulation.py                           # viewer: click actor, then target
    python scripts/06_run_simulation.py --auto                    # viewer: agent -> nearest object
    python scripts/06_run_simulation.py --no-view                 # headless agent -> nearest object
    python scripts/06_run_simulation.py --actor=ROI_39 --target=ROI_11   # object vs object
    python scripts/06_run_simulation.py --start=1.0,0.5,1.2       # agent start point (m)
"""

import json
import os
import sys

import numpy as np
import mujoco

# src/Module4-Physic_Engine/ (holds preprocessing/ and simulation/)
ENGINE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)


from preprocessing.roi_config import ALL_ROIS, distance_to_aabb, is_thin_roi
from simulation.topology_guard import TopologyGuard

PROBE_BODY = "probe"
TASK_REPORT_NAME = "task_report.json"
PARKED_PROBE = np.array([0.0, 0.0, 100.0])

# Actor motion
AGENT_HALF_SIZE = np.array([0.06, 0.06, 0.06])  # m
START_HEIGHT = 1.2          # m above the lowest ROI, default agent start height (above tables)
STANDOFF = 0.10             # m gap between actor and target AABB before the push (+ braking distance)
BRAKE_MARGIN = 1.3          # safety factor on the braking distance down to PUSH_SPEED
SAFE_MAX_SPEED = 15.0       # m/s, speeds are capped here (the Topology Guard flags > 20 m/s as an explosion)
ACCEL_FRACTION = 0.8        # an object actor accelerates / brakes at this fraction of MAX_PUSH_FORCE / mass
TRAVEL_SPEED = 40.0          # m/s, actor 1 on its way to actor 2
PUSH_SPEED = 1.5            # m/s, speed at the impact
PUSH_DEPTH = 0.20           # m pushed after the first contact
PUSH_OVERSHOOT = 0.10       # m past the target centre (+ its half size) before giving up: MISSED
MAX_PUSH_FORCE = 2500.0      # N, the actor stops pushing above this (a kinematic body has no limit) ...
FORCE_WINDOW = 0.05         # s ... averaged over this window, so the impact spike alone does not stop it

# Object actor: PD tracking of the moving set point
ACTOR_OMEGA = 15.0          # rad/s, natural frequency of the position / orientation PD
MAX_ACTOR_TORQUE = 50.0     # N.m
ACTOR_MAX_LAG = 0.08        # m, the set point waits while the object lags further behind
STALL_TIME = 2.0            # s lagging before the actor is declared stuck
STALL_SPEED_FRACTION = 0.2  # lagging below this fraction of the set point speed = stalled
PRESS_TIME = 0.5            # s pressing at the force cap against the target before it counts as BLOCKED
FLOOR_MARGIN_OBJECT = 0.06  # m between an object actor's AABB bottom and the floor while pushing
LIFT_OFF = 0.06             # m an object actor rises straight up before its straight line
# A small (but solid) actor keeps every other parameter but moves and pushes more gently:
# TRAVEL_SPEED, PUSH_SPEED and MAX_PUSH_FORCE are scaled by size / SMALL_OBJECT_SIZE.
SMALL_OBJECT_SIZE = 0.40    # m, largest box edge from which an actor counts as full size
SMALL_MIN_SCALE = 0.3       # the scale never goes below this (tiny objects)
STEP_UP = 0.05              # m an object actor rises when it snags on something before the contact ...
MAX_STEP_UPS = 4            # ... at most this many times
STEP_UP_AFTER = 0.2         # s of lagging that counts as snagged

# Timeline
SETTLE_TIME = 0.5           # s before the actor starts
HOLD_TIME = 0.3             # s holding at the end of the push
OBSERVE_MIN = 0.5           # s watching the scene after the retreat / release ...
OBSERVE_MAX = 3.0           # ... until everything is at rest, at most this long
MAX_TASK_TIME = 30.0        # s, headless safety stop

# Analysis
REST_SPEED = 0.02           # m/s, a body slower than this is at rest
TOPPLE_ANGLE_DEG = 45.0
MOVED_EPS = 0.01            # m, displacement counted as "moved"
N_PUSH_HEIGHTS = 7          # aim heights tried, spread over the target AABB
# mj_ray geom groups: 0 = probe / plane, 3 = collision hulls (visual meshes are group 1).
RAY_GROUPS = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)

WAIT, SETTLE, APPROACH, PUSH, HOLD, RETREAT, OBSERVE, DONE = (
    "WAIT", "SETTLE", "APPROACH", "PUSH", "HOLD", "RETREAT", "OBSERVE", "DONE")


def default_start():
    """Middle of the room (centre of all ROI boxes), START_HEIGHT above the lowest ROI."""
    mins = np.asarray([r["min"] for r in ALL_ROIS.values()], dtype=np.float64)
    maxs = np.asarray([r["max"] for r in ALL_ROIS.values()], dtype=np.float64)
    start = ((mins + maxs) / 2.0).mean(axis=0)
    start[2] = mins[:, 2].min() + START_HEIGHT
    return start


def parse_start(text):
    values = [float(v) for v in text.replace(" ", "").split(",")]
    if len(values) != 3:
        raise ValueError(f"--start needs x,y,z, got '{text}'")
    return np.asarray(values)


def agent_spec(start=None):
    """Probe description for build_scene / build_unified_xml."""
    return {"pos": default_start() if start is None else np.asarray(start, dtype=np.float64),
            "half_size": AGENT_HALF_SIZE}


def _support(direction, half):
    """Half width of an axis-aligned box along a direction."""
    return float(np.abs(direction) @ np.asarray(half))


def _category(name):
    return str(ALL_ROIS[name].get("category", "unknown")) if name in ALL_ROIS else "agent"


class AutonomousTask:
    """Drives the actor through the task. Call `before_step()` / `after_step()` around every
    TopologyGuard.step(); the report is in `self.report` once `phase == DONE`.
    With `wait_for_click` nothing happens until `select()` has been given an actor and a target."""

    def __init__(self, model, data, start=None, target=None, actor=None, guard=None, log=print,
                 wait_for_click=False):
        self.model, self.data, self.guard, self.log = model, data, guard, log
        self.probe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PROBE_BODY)
        if self.probe_id < 0:
            raise RuntimeError("Scene has no probe body; rebuild it with the agent (--rebuild)")
        self.mocap = int(model.body_mocapid[self.probe_id])
        self.home = default_start() if start is None else np.asarray(start, dtype=np.float64)
        self.initial_target = target
        self.initial_actor = actor
        self.wait_for_click = wait_for_click and target is None and actor is None
        self.dt = float(model.opt.timestep)
        # Free body id -> (qpos adr, dof adr)
        self.free = {
            int(model.jnt_bodyid[j]): (int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))
            for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        }
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor_plane")
        self.floor_z = float(model.geom_pos[floor][2]) if floor >= 0 else \
            min(float(r["min"][2]) for r in ALL_ROIS.values())
        self.excluded = {int(s) for s in model.exclude_signature}
        if actor is not None:
            self._check_actor(actor, raise_error=True)
        self._steps = 0
        self.actor_id = None
        self.reset()

    def _name(self, body):
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(body)) or f"body_{body}"

    def _body(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _check_actor(self, name, raise_error=False):
        if name in ALL_ROIS and self._body(name) in self.free:
            return True
        msg = (f"{name} ({_category(name)}) is static (welded to the world) and cannot move"
               if name in ALL_ROIS else f"{name} is not an ROI object of this scene")
        if raise_error:
            raise RuntimeError(msg)
        self.log(f"[Task] {msg} - double-click a movable object as the actor")
        return False

    # -- run setup -----------------------------------------------------------------------
    def reset(self):
        """Back to the initial setup (also after Backspace in the viewer)."""
        self._release()
        self.pending_actor = None
        self._set_actor(self.initial_actor)
        if self.actor_id in (None, self.probe_id):
            self.start = self.home.copy()
            if self.actor_id == self.probe_id:
                self.data.mocap_pos[self.mocap] = self.start
        else:
            self.start = self._actor_pos()
        self.requested_target = self.initial_target
        self.selected_by = "argument" if self.initial_target else "nearest"
        self._clear_run(WAIT if self.wait_for_click else SETTLE)

    def _set_actor(self, name):
        """name None -> the kinematic agent; otherwise an ROI object carried by forces."""
        if name is None and not self.wait_for_click:
            self.actor_name, self.actor_id = "agent", self.probe_id
            self.half = AGENT_HALF_SIZE.copy()
            self.below = self.above = float(AGENT_HALF_SIZE[2])
            return
        if name is None:            # waiting for clicks: no actor yet
            self.actor_name, self.actor_id, self.half = None, None, None
        else:
            roi = ALL_ROIS[name]
            self.actor_name, self.actor_id = name, self._body(name)
            self.half = (np.asarray(roi["max"], dtype=np.float64) - np.asarray(roi["min"], dtype=np.float64)) / 2.0
            # Body origin = vertex centroid, not the AABB centre: keep the real reach below / above it.
            origin_z = float(self.model.body_pos[self.actor_id][2])
            self.below = origin_z - float(roi["min"][2])
            self.above = float(roi["max"][2]) - origin_z
        self.data.mocap_pos[self.mocap] = PARKED_PROBE

    def _clear_run(self, phase):
        self.phase = phase
        self.phase_start = float(self.data.time)
        self.timeline = [(phase, self.phase_start)]
        self.plan = None
        self.target_id = None
        self.waypoints = []
        self.initial = None
        self.setpoint = None
        self.setpoint_vel = np.zeros(3)
        self.sp_speed = 0.0         # current set point speed (ramps with self.accel)
        self.travel_speed, self.push_speed, self.max_force = TRAVEL_SPEED, PUSH_SPEED, MAX_PUSH_FORCE
        self.size_scale, self.size_class = 1.0, "agent"
        self.line_start = None
        self.accel = np.inf
        self.driving = False
        self.stall = 0.0
        self.step_ups = 0
        self.stuck = False
        self.blocked_by = []
        self.first_contact = None
        self.push_stop = None
        self.force_limited = False
        self.contact_points = []
        self.force_trace = []       # (time, summed normal force actor -> target)
        self.path_hits = {}         # non-target bodies the actor touched: steps in contact
        self.actor_max_depth = 0.0
        self.report = None

    def select(self, name):
        """Viewer double-click: the first object becomes the actor, the next one the target."""
        if name not in ALL_ROIS:
            return
        if self.pending_actor is None or self.pending_actor == name:
            if self.pending_actor == name or not self._check_actor(name):
                return
            if self.phase not in (WAIT, DONE):
                self.log("[Task] new actor selected: aborting the current run")
                self._release()
                self._clear_run(WAIT)
            self.pending_actor = name
            self.log(f"[Task] actor {name} ({_category(name)}) selected -> double-click the object it should hit")
            return
        actor, self.pending_actor = self.pending_actor, None
        self.start_run(actor, name, selected_by="click")

    def start_run(self, actor, target, selected_by="argument"):
        """Start a run now: `actor` (ROI name, None = agent) moves into `target` (ROI name)."""
        self._release()
        self._set_actor(actor)
        if actor is None:
            self.actor_name, self.actor_id, self.half = "agent", self.probe_id, AGENT_HALF_SIZE.copy()
            self.below = self.above = float(AGENT_HALF_SIZE[2])
            self.data.mocap_pos[self.mocap] = self.home
        self.start = self._actor_pos()
        self.requested_target = target
        self.selected_by = selected_by
        self._clear_run(SETTLE)
        self._begin()

    def _begin(self):
        if self.actor_id != self.probe_id and self.guard is not None:
            self.guard.wake(self.actor_id)
        self.start = self._actor_pos()
        self._plan()
        if self.guard is not None:
            # The actor is driven on purpose and the target must react to the impact at once:
            # neither gets the "just woken" speed clamp (bodies hit in a chain reaction still do).
            self.guard.unclamped = {self.actor_id, self.target_id}
        d = self.data
        bodies = set(self.free) | {self.target_id}
        self.initial = {b: (d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()) for b in bodies}
        self.setpoint = self.start.copy()
        self.hold_quat = d.xquat[self.actor_id].copy()
        self.driving = True
        self._enter(APPROACH)

    # -- planning ------------------------------------------------------------------------
    def _aabb(self, name):
        """ROI AABB moved with its body since the scan (objects may have been pushed before)."""
        roi = ALL_ROIS[name]
        body = self._body(name)
        shift = self.data.xpos[body] - self.model.body_pos[body]
        return np.asarray(roi["min"], dtype=np.float64) + shift, np.asarray(roi["max"], dtype=np.float64) + shift

    def _candidates(self):
        """Dynamic ROI objects (nearest first), the actor itself excluded."""
        cands = []
        for body in self.free:
            name = self._name(body)
            if name not in ALL_ROIS or body == self.actor_id:
                continue
            lo, hi = self._aabb(name)
            cands.append({"id": name, "body": body, "category": _category(name),
                          "distance_m": distance_to_aabb(self.start, lo, hi)})
        return sorted(cands, key=lambda c: c["distance_m"])

    def _ray(self, origin, vec):
        geomid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(self.model, self.data, np.asarray(origin, dtype=np.float64),
                             np.asarray(vec, dtype=np.float64), RAY_GROUPS, 1, self.actor_id, geomid)
        if dist < 0:
            return np.inf, -1
        return float(dist), int(self.model.geom_bodyid[geomid[0]])

    def _push_heights(self, lo, hi):
        """Candidate actor-centre heights facing the target AABB, preferred height first."""
        # An object actor keeps clear of rugs / scan bumps: scraping the floor stalls it.
        margin, ground = 0.02, self.floor_z
        if self.actor_id != self.probe_id:
            # The scanned floor / rugs lie above the floor plane: use the surface the actor stands
            # on when it is on the floor (not when it starts on a table).
            margin = FLOOR_MARGIN_OBJECT
            ground = max(self.floor_z, min(self.start[2] - self.below, self.floor_z + 0.10))
        low = max(lo[2] + min(self.below, (hi[2] - lo[2]) / 2.0), ground + self.below + margin)
        high = max(hi[2] - min(self.above, (hi[2] - lo[2]) / 2.0), low)
        # Agent: aim at the target centre. Object actor: stay as level as possible (closest to
        # its own height), so the straight line does not climb up a wall.
        prefer = (lo[2] + hi[2]) / 2.0 if self.actor_id == self.probe_id else self.line_start[2]
        center = min(max(prefer, low), high)
        levels = np.unique(np.append(np.linspace(low, high, N_PUSH_HEIGHTS), center))
        return sorted(levels, key=lambda z: abs(z - center))

    def _aim(self, target, lo, hi):
        """Aim point inside the target box, on a straight line from the actor: the first
        height (closest to the target centre) where a ray from the actor hits the target
        before anything else. Returns (aim point, distance to the target surface or None)."""
        center = (lo + hi) / 2.0
        heights = self._push_heights(lo, hi)
        for z in heights:
            aim = np.array([center[0], center[1], z])
            vec = aim - self.line_start
            if np.linalg.norm(vec) < 1e-6:
                continue
            dist, body = self._ray(self.line_start, vec / np.linalg.norm(vec))
            if body == target:
                return aim, dist
        return np.array([center[0], center[1], heights[0]]), None

    def _collision_geoms(self, body):
        m = self.model
        return [g for g in range(m.ngeom) if m.geom_bodyid[g] == body
                and (m.geom_group[g] == 3 or body == self.probe_id)]

    def _brake_gap(self):
        """Gap at which the actor must start braking to hit at PUSH_SPEED (v^2 - v0^2 = 2 a d)."""
        if not np.isfinite(self.accel):
            return STANDOFF
        return STANDOFF + BRAKE_MARGIN * max(self.sp_speed ** 2 - self.push_speed ** 2, 0.0) / (2.0 * self.accel)

    def _gap(self, cap):
        """Smallest distance between the actor's and the target's collision hulls, capped at
        `cap`. Runs every step, so the far pairs are dropped with one numpy bounding-sphere
        test and the exact mj_geomDistance only runs on the few closest pairs."""
        m, d = self.model, self.data
        a, b = self.actor_geoms, self.target_geoms
        rough = (np.linalg.norm(d.geom_xpos[a][:, None, :] - d.geom_xpos[b][None, :, :], axis=2)
                 - m.geom_rbound[a][:, None] - m.geom_rbound[b][None, :])
        best = cap
        for flat in np.argsort(rough, axis=None):
            i, j = divmod(int(flat), len(b))
            if rough[i, j] >= best:
                break               # sorted: every remaining pair is further than `best`
            best = min(best, mujoco.mj_geomDistance(m, d, a[i], b[j], best, None))
        return float(best)

    def _plan(self):
        cands = self._candidates()
        if self.requested_target:
            name = self.requested_target
            if name not in ALL_ROIS or self._body(name) < 0:
                raise RuntimeError(f"Target {name} is not an ROI object of this scene")
            if self._body(name) == self.actor_id:
                raise RuntimeError("Actor and target are the same object")
            lo, hi = self._aabb(name)
            chosen = {"id": name, "body": self._body(name), "category": _category(name),
                      "distance_m": distance_to_aabb(self.start, lo, hi)}
        else:
            if not cands:
                raise RuntimeError("No dynamic ROI object in the scene to interact with")
            # "unknown" segments are usually scan scraps: only fall back to them when nothing else.
            labelled = [c for c in cands if c["category"] != "unknown"] or cands
            chosen = labelled[0]

        lo, hi = self._aabb(chosen["id"])
        half = (hi - lo) / 2.0
        # An object actor first lifts LIFT_OFF straight up: scanned objects sit a little inside
        # the scanned floor and would otherwise grind along it at a few cm/s.
        object_actor = self.actor_id != self.probe_id
        self.line_start = self.start + np.array([0.0, 0.0, LIFT_OFF if object_actor else 0.0])
        aim, hit = self._aim(chosen["body"], lo, hi)
        clear = hit is not None
        line = aim - self.line_start
        direction = line / max(np.linalg.norm(line), 1e-9)
        # Straight line: full speed until just before contact, push speed after. The ray hit
        # gives the target surface; without one (blocked line of sight) use the target box.
        surface = hit if clear else float(np.linalg.norm(line)) - _support(direction, half)
        pre = self.line_start + direction * max(surface - _support(direction, self.half) - STANDOFF, 0.0)
        # An object actor speeds up / brakes at what its force cap allows; the agent instantly.
        # Braking starts from the hull-to-hull gap (see _brake_gap); `pre` is only a fallback.
        if object_actor:
            size = float(2.0 * self.half.max())
            if is_thin_roi(ALL_ROIS[self.actor_name]):
                self.size_class = "soft"
            elif size < SMALL_OBJECT_SIZE:
                self.size_class = "small"
                self.size_scale = max(size / SMALL_OBJECT_SIZE, SMALL_MIN_SCALE)
            else:
                self.size_class = "large"
            k = self.size_scale
            self.travel_speed, self.push_speed, self.max_force = TRAVEL_SPEED * k, PUSH_SPEED * k, MAX_PUSH_FORCE * k
            self.accel = ACCEL_FRACTION * self.max_force / float(self.model.body_subtreemass[self.actor_id])
            if self.size_class == "small":
                self.log(f"[Task] {self.actor_name} is small ({size:.2f} m): speed and force x{k:.2f}")
        if self.travel_speed > SAFE_MAX_SPEED:
            self.log(f"[Task] travel speed {self.travel_speed:g} m/s capped at {SAFE_MAX_SPEED:g} m/s")
        self.actor_geoms = np.asarray(self._collision_geoms(self.actor_id), dtype=int)
        self.target_geoms = np.asarray(self._collision_geoms(chosen["body"]), dtype=int)
        push_end = aim + direction * (_support(direction, half) + PUSH_OVERSHOOT)
        self.target_id = chosen["body"]
        self.target_static = chosen["body"] not in self.free
        self.waypoints = [self.line_start, pre] if object_actor else [pre]
        a, b = sorted((self.actor_id, self.target_id))
        contact_disabled = ((a << 16) + b) in self.excluded or ((b << 16) + a) in self.excluded
        self.plan = {
            "target": chosen["id"],
            "category": chosen["category"],
            "distance_m": chosen["distance_m"],
            "selected_by": self.selected_by,
            "candidates": [{k: c[k] for k in ("id", "category", "distance_m")} for c in cands[:5]],
            "direction": direction,
            "line_of_sight_clear": clear,
            "aim_point": aim,
            "pre_contact": pre,
            "push_end": push_end,
            "contact_disabled": contact_disabled,
        }
        self.log(f"[Task] {self.actor_name} -> {chosen['id']} ({chosen['category']}"
                 + (", static" if self.target_static else "") + f"), {chosen['distance_m']:.2f} m away; "
                 "straight line"
                 + ("" if clear else "  [something else is in the line of sight, the actor may hit it first]"))
        if contact_disabled:
            self.log(f"[Task] WARNING: contact {self.actor_name} <-> {chosen['id']} is excluded in the scene "
                     "(they overlapped in the scan): they will pass through each other")

    # -- motion --------------------------------------------------------------------------
    def _enter(self, phase):
        self.phase = phase
        self.phase_start = float(self.data.time)
        self.stall = 0.0
        self.setpoint_vel = np.zeros(3)
        if phase != PUSH:           # APPROACH -> PUSH keeps its speed and brakes smoothly
            self.sp_speed = 0.0
        self.timeline.append((phase, self.phase_start))
        self.log(f"[Task] t={self.data.time:6.2f}s  {phase}")

    def _actor_pos(self):
        if self.actor_id == self.probe_id:
            return self.data.mocap_pos[self.mocap].copy()
        return self.data.xpos[self.actor_id].copy()

    def _move_toward(self, goal, speed):
        """Advance the set point towards `goal`; True when it is there."""
        if self.actor_id != self.probe_id:
            # An object that lags behind makes the set point wait for it. It only counts as
            # stalled (blocked, wedged) when it is also slow; while it is still accelerating
            # the feedforward velocity is kept, so it does not stop and go.
            if np.linalg.norm(self.setpoint - self._actor_pos()) > ACTOR_MAX_LAG:
                _, da = self.free[self.actor_id]
                actual = float(np.linalg.norm(self.data.qvel[da:da + 3]))
                if actual < STALL_SPEED_FRACTION * max(self.sp_speed, self.push_speed):
                    self.stall += self.dt
                    self.setpoint_vel = np.zeros(3)
                    self.sp_speed = actual
                return False
            self.stall = 0.0
        # Speed profile: ramp towards `speed` at self.accel (instant for the agent).
        speed = min(speed, SAFE_MAX_SPEED)
        if not np.isfinite(self.accel):
            self.sp_speed = speed
        elif self.sp_speed < speed:
            self.sp_speed = min(speed, self.sp_speed + self.accel * self.dt)
        else:
            self.sp_speed = max(speed, self.sp_speed - self.accel * self.dt)
        delta = goal - self.setpoint
        dist = float(np.linalg.norm(delta))
        step = self.sp_speed * self.dt
        if dist <= step:
            self.setpoint = np.asarray(goal, dtype=np.float64).copy()
            self.setpoint_vel = np.zeros(3)
            return True
        self.setpoint = self.setpoint + delta / dist * step
        self.setpoint_vel = delta / dist * self.sp_speed
        return False

    def _follow(self, speed):
        if self.waypoints and self._move_toward(self.waypoints[0], speed):
            self.waypoints.pop(0)
            self.sp_speed = 0.0     # corner between two straight segments: start again from rest
        return not self.waypoints

    def _drive(self):
        """Apply the set point: mocap pose for the agent, PD force/torque for an object."""
        d, m = self.data, self.model
        if self.actor_id == self.probe_id:
            d.mocap_pos[self.mocap] = self.setpoint
            return
        b = self.actor_id
        _, da = self.free[b]
        mass = float(m.body_subtreemass[b])
        rot = d.xmat[b].reshape(3, 3)
        # Velocity feedforward: damp against the set point velocity, so the object keeps up at speed.
        force = (mass * ACTOR_OMEGA ** 2 * (self.setpoint - d.xpos[b])
                 + 2.0 * mass * ACTOR_OMEGA * (self.setpoint_vel - d.qvel[da:da + 3]))
        norm = np.linalg.norm(force)
        if norm > self.max_force:
            force *= self.max_force / norm
        inertia = float(m.body_inertia[b].max())
        q_err, q_inv, rotvec = np.zeros(4), np.zeros(4), np.zeros(3)
        mujoco.mju_negQuat(q_inv, d.xquat[b])
        mujoco.mju_mulQuat(q_err, self.hold_quat, q_inv)
        mujoco.mju_quat2Vel(rotvec, q_err, 1.0)
        torque = inertia * ACTOR_OMEGA ** 2 * rotvec - 2.0 * inertia * ACTOR_OMEGA * (rot @ d.qvel[da + 3:da + 6])
        torque = np.clip(torque, -MAX_ACTOR_TORQUE, MAX_ACTOR_TORQUE)
        d.xfrc_applied[b, :3] = force - mass * m.opt.gravity
        d.xfrc_applied[b, 3:] = torque

    def _release(self):
        """Stop driving the actor; an object then falls / settles under gravity alone."""
        self.driving = False
        if self.actor_id is not None and self.actor_id != self.probe_id:
            self.data.xfrc_applied[self.actor_id] = 0.0

    def _touching(self, body):
        m, d = self.model, self.data
        names = set()
        for i in range(d.ncon):
            b1, b2 = int(m.geom_bodyid[d.contact[i].geom[0]]), int(m.geom_bodyid[d.contact[i].geom[1]])
            if body in (b1, b2):
                names.add(self._name(b2 if b1 == body else b1))
        return sorted(names)

    def _at_rest(self):
        return all(np.linalg.norm(self.data.qvel[da:da + 3]) < REST_SPEED for _, da in self.free.values())

    def before_step(self):
        """Advance the task state machine and drive the actor for the next step."""
        d = self.data
        if d.time <= 0.0 and self._steps > 0:   # viewer reset (Backspace)
            self.log("[Task] scene reset -> back to the start")
            self.reset()
        self._steps += 1
        if self.actor_id is None or self.actor_id != self.probe_id:
            d.mocap_pos[self.mocap] = PARKED_PROBE
        t = float(d.time)
        elapsed = t - self.phase_start
        object_actor = self.actor_id is not None and self.actor_id != self.probe_id

        if self.phase == WAIT:
            return
        if self.phase == SETTLE:
            if elapsed >= SETTLE_TIME:
                self._begin()
        elif self.phase == APPROACH:
            # Slow down to PUSH_SPEED once the hulls are within the braking distance, so the
            # impact happens at PUSH_SPEED whatever the shape of the target.
            brake_gap = self._brake_gap()
            if self._follow(self.travel_speed) or self._gap(brake_gap + 1.0) <= brake_gap:
                self.waypoints = []
                self._enter(PUSH)
        elif self.phase == PUSH:
            goal = self.push_stop if self.push_stop is not None else self.plan["push_end"]
            if self.force_limited:
                self._enter(HOLD)
            elif self._move_toward(goal, self.push_speed):
                if self.first_contact is None:
                    self.log("[Task] reached the far side without touching the target -> MISSED")
                self._enter(HOLD)
            elif (object_actor and self.first_contact is None and self.stall > STEP_UP_AFTER
                  and self.step_ups < MAX_STEP_UPS):
                # Snagged on a rug edge / scan bump before reaching the target: rise and go on.
                self.step_ups += 1
                self.setpoint[2] += STEP_UP
                self.plan["push_end"][2] += STEP_UP
                self.stall = 0.0
                self.log(f"[Task] t={d.time:6.2f}s  {self.actor_name} snagged -> rising {STEP_UP * 100:.0f} cm")
            elif object_actor and self.stall > PRESS_TIME and self.first_contact is not None:
                # Pressing at the force cap without moving: the target resists.
                self.force_limited = True
                self.log(f"[Task] t={d.time:6.2f}s  {self.actor_name} cannot push further "
                         f"(force cap {self.max_force:.0f} N)")
                self._enter(HOLD)
        elif self.phase == HOLD:
            if elapsed >= HOLD_TIME:
                if object_actor:
                    self._release()
                    self._enter(OBSERVE)
                else:
                    self.waypoints = [self.plan["pre_contact"], self.start]
                    self._enter(RETREAT)
        elif self.phase == RETREAT:
            if self._follow(self.travel_speed):
                self._enter(OBSERVE)
        elif self.phase == OBSERVE:
            if elapsed >= OBSERVE_MIN and (self._at_rest() or elapsed >= OBSERVE_MAX):
                self.finish()

        if self.phase in (APPROACH, PUSH, HOLD) and self.stall > STALL_TIME:
            self.stuck = True
            self.blocked_by = self._touching(self.actor_id)
            self.log(f"[Task] t={d.time:6.2f}s  {self.actor_name} is stuck on the way"
                     + (f" (blocked by {', '.join(self.blocked_by)})" if self.blocked_by else "") + " -> released")
            self._release()
            self._enter(OBSERVE)
        if self.driving and self.phase in (APPROACH, PUSH, HOLD, RETREAT):
            self._drive()

    def after_step(self):
        """Record the actor contacts of the step just simulated."""
        if self.target_id is None or self.phase in (DONE, WAIT):
            return
        m, d = self.model, self.data
        actor = self.actor_id
        object_actor = actor != self.probe_id
        force = np.zeros(6)
        f_target = 0.0
        for i in range(d.ncon):
            con = d.contact[i]
            b1, b2 = int(m.geom_bodyid[con.geom[0]]), int(m.geom_bodyid[con.geom[1]])
            if actor not in (b1, b2):
                continue
            other = b2 if b1 == actor else b1
            if other != self.target_id:
                # An object actor slides over the floor and room scan: those are not path collisions.
                if self.phase != OBSERVE and (not object_actor or m.body_weldid[other] != 0):
                    self.actor_max_depth = max(self.actor_max_depth, -float(con.dist))
                    name = self._name(other)
                    self.path_hits[name] = self.path_hits.get(name, 0) + 1
                continue
            self.actor_max_depth = max(self.actor_max_depth, -float(con.dist))
            mujoco.mj_contactForce(m, d, i, force)
            f_target += abs(float(force[0]))
            self.contact_points.append(np.array(con.pos))
            if self.first_contact is None:
                pos = self.setpoint.copy()
                if object_actor:
                    vel = d.qvel[self.free[actor][1]:self.free[actor][1] + 3]
                else:
                    vel = self.setpoint_vel
                self.first_contact = {"time_s": float(d.time), "phase": self.phase,
                                      "actor_pos": self._actor_pos(), "point": np.array(con.pos),
                                      "speed": float(np.dot(vel, self.plan["direction"]))}
                self.push_stop = pos + self.plan["direction"] * PUSH_DEPTH
                self.log(f"[Task] t={d.time:6.2f}s  {self.actor_name} hits {self._name(other)}, "
                         f"pushing {PUSH_DEPTH:.2f} m more")
        self.force_trace.append((float(d.time), f_target))
        window = max(1, int(round(FORCE_WINDOW / self.dt)))
        f_avg = sum(f for _, f in self.force_trace[-window:]) / window
        # The agent has no force limit of its own; an object actor is capped by its PD force.
        if not object_actor and self.phase == PUSH and f_avg > self.max_force and not self.force_limited:
            self.force_limited = True
            self.push_stop = self.setpoint.copy()
            self.log(f"[Task] t={d.time:6.2f}s  push force {f_avg:.0f} N > {self.max_force:.0f} N -> actor stops")

    # -- analysis ------------------------------------------------------------------------
    def _pose_change(self, body):
        p0, r0 = self.initial[body]
        p1, r1 = self.data.xpos[body].copy(), self.data.xmat[body].reshape(3, 3).copy()
        disp = p1 - p0
        tilt = float(np.degrees(np.arccos(np.clip(r0[:, 2] @ r1[:, 2], -1.0, 1.0))))
        rel = r1 @ r0.T
        return {
            "initial_pos": p0.tolist(),
            "final_pos": p1.tolist(),
            "displacement_m": disp.tolist(),
            "horizontal_displacement_m": float(np.linalg.norm(disp[:2])),
            "vertical_displacement_m": float(disp[2]),
            "tilt_deg": tilt,
            "yaw_deg": float(np.degrees(np.arctan2(rel[1, 0], rel[0, 0]))),
            "toppled": tilt > TOPPLE_ANGLE_DEG,
        }

    def finish(self, timed_out=False):
        self._release()
        tgt, actor = self.target_id, self.actor_id
        object_actor = actor != self.probe_id
        response = self._pose_change(tgt)
        others = {}
        for b, (q0, _) in self.initial.items():
            moved = float(np.linalg.norm(self.data.xpos[b] - q0))
            if b not in (tgt, actor) and moved > MOVED_EPS:
                others[self._name(b)] = moved

        forces = np.asarray([f for _, f in self.force_trace]) if self.force_trace else np.zeros(1)
        contact_steps = int(np.count_nonzero(forces > 0))
        horizontal, tilt = response["horizontal_displacement_m"], response["tilt_deg"]
        if self.first_contact is None:
            outcome = "MISSED"
            meaning = (("the actor got stuck on the way"
                        + (f", blocked by {', '.join(self.blocked_by)}" if self.blocked_by else "")
                        + (" (environment = unsegmented scan geometry)" if "environment" in self.blocked_by else ""))
                       if self.stuck else
                       "the actor never touched the target (path blocked, contact excluded, or it passed beside it)")
        elif self.target_static:
            outcome = "IMPACT"
            meaning = "the actor hits a fixed structure (the target cannot move)"
        elif tilt > TOPPLE_ANGLE_DEG:
            outcome = "TOPPLED"
            meaning = f"the hit tips the target over ({tilt:.0f} deg)"
        elif horizontal < MOVED_EPS or (self.force_limited and horizontal < PUSH_DEPTH / 2):
            outcome = "BLOCKED"
            meaning = (f"the target resists the push (moved {horizontal:.2f} m"
                       + (f", needs more than {self.max_force:.0f} N" if self.force_limited else "")
                       + "): too heavy, wedged or held by its neighbours")
        else:
            outcome = "PUSHED"
            meaning = f"the target slides {horizontal:.2f} m and stays upright"
        actor_response = self._pose_change(actor) if object_actor else None
        if actor_response and actor_response["toppled"]:
            meaning += f"; the actor itself falls over ({actor_response['tilt_deg']:.0f} deg)"

        plan = self.plan
        self.report = {
            "outcome": outcome,
            "prediction": meaning,
            "timed_out": timed_out,
            "actor": {
                "id": self.actor_name,
                "type": "object" if object_actor else "agent",
                "category": _category(self.actor_name) if object_actor else "agent",
                "mass_kg": float(self.model.body_subtreemass[actor]) if object_actor else None,
                "thin": bool(object_actor and is_thin_roi(ALL_ROIS[self.actor_name])),
                "start": self.start.tolist(),
                "half_size_m": self.half.tolist(),
                "size_class": self.size_class,
                "size_scale": self.size_scale,
                "travel_speed_mps": self.travel_speed,
                "push_speed_mps": self.push_speed,
                "push_depth_m": PUSH_DEPTH,
                "max_push_force_N": self.max_force,
                "stuck": self.stuck,
                "blocked_by": self.blocked_by,
                "response": actor_response,
            },
            "target_selection": {
                "target": plan["target"],
                "category": plan["category"],
                "static": self.target_static,
                "thin": is_thin_roi(ALL_ROIS[plan["target"]]),
                "selected_by": plan["selected_by"],
                "distance_m": plan["distance_m"],
                "nearest_candidates": plan["candidates"],
            },
            "path": {
                "trajectory": "straight line",
                "direction": plan["direction"].tolist(),
                "line_of_sight_clear": plan["line_of_sight_clear"],
                "contact_disabled_in_scene": plan["contact_disabled"],
                "aim_point": plan["aim_point"].tolist(),
                "pre_contact": plan["pre_contact"].tolist(),
                "other_bodies_touched_by_actor": self.path_hits,
            },
            "timeline_s": [[phase, t] for phase, t in self.timeline],
            "contact": None if self.first_contact is None else {
                "first_contact_time_s": self.first_contact["time_s"],
                "first_contact_phase": self.first_contact["phase"],
                "first_contact_point": self.first_contact["point"].tolist(),
                "impact_speed_mps": self.first_contact["speed"],
                "contact_steps": contact_steps,
                "contact_duration_s": contact_steps * self.dt,
                "peak_normal_force_N": float(forces.max()),
                "mean_normal_force_N": float(forces[forces > 0].mean()) if contact_steps else 0.0,
                "impulse_Ns": float(forces.sum() * self.dt),
                "force_limited": self.force_limited,
                "actor_max_penetration_m": self.actor_max_depth,
            },
            "target_response": response,
            "side_effects": {
                "other_bodies_moved_m": dict(sorted(others.items(), key=lambda kv: -kv[1])),
            },
            "simulated_time_s": float(self.data.time),
        }
        if self.guard is not None:
            self.report["topology_guard"] = self.guard.report()
        self._enter(DONE)
        return self.report


def save_report(report, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, TASK_REPORT_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


def print_summary(report):
    sel, resp, contact, actor = (report["target_selection"], report["target_response"], report["contact"],
                                 report["actor"])
    print("\n" + "=" * 70)
    print(f"SIMULATION RESULT: {report['outcome']}  -  {report['prediction']}")
    print("=" * 70)
    if actor["type"] == "object":
        print(f"  actor            {actor['id']} ({actor['category']}, {actor['mass_kg']:.1f} kg, "
              f"{actor['size_class']}, speed/force x{actor['size_scale']:.2f})")
    else:
        print("  actor            agent (kinematic probe)")
    print(f"  target           {sel['target']} ({sel['category']}{', static' if sel['static'] else ''}), "
          f"{sel['distance_m']:.2f} m from the actor [{sel['selected_by']}]")
    if contact:
        print(f"  first contact    t={contact['first_contact_time_s']:.2f}s at "
              + ", ".join(f"{v:.3f}" for v in contact["first_contact_point"])
              + f", impact speed {contact['impact_speed_mps']:.2f} m/s")
        print(f"  contact force    peak {contact['peak_normal_force_N']:.1f} N, "
              f"mean {contact['mean_normal_force_N']:.1f} N, impulse {contact['impulse_Ns']:.2f} N.s, "
              f"{contact['contact_duration_s']:.2f} s")
    print(f"  target moved     {resp['horizontal_displacement_m']:.3f} m horizontal, "
          f"{resp['vertical_displacement_m']:+.3f} m vertical, tilt {resp['tilt_deg']:.1f} deg, "
          f"yaw {resp['yaw_deg']:+.1f} deg")
    if actor["response"]:
        a = actor["response"]
        print(f"  actor ended      {a['horizontal_displacement_m']:.3f} m from its start, tilt {a['tilt_deg']:.1f} deg")
    others = report["side_effects"]["other_bodies_moved_m"]
    print(f"  side effects     {len(others)} other objects moved"
          + (": " + ", ".join(f"{n} {v:.2f} m" for n, v in list(others.items())[:5]) if others else ""))
    hits = report["path"]["other_bodies_touched_by_actor"]
    if hits:
        print(f"  actor also hit   {', '.join(hits)}")
    if report["path"]["contact_disabled_in_scene"]:
        print("  WARNING          contact actor <-> target is excluded in this scene")
    guard = report.get("topology_guard")
    if guard:
        print(f"  topology guard   {guard['fault_count']} faults")
    if report["timed_out"]:
        print("  (stopped at the time limit before the scene came to rest)")


def run_headless(xml_path, out_dir, sleep=True, start=None, target=None, actor=None, max_time=MAX_TASK_TIME):
    """Run the task without a viewer; returns the report (also saved to out_dir)."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    guard = TopologyGuard(model, data, sleep=sleep)
    task = AutonomousTask(model, data, start=start, target=target, actor=actor, guard=guard)
    guard.start()
    while task.phase != DONE and data.time < max_time:
        task.before_step()
        if task.phase == DONE:
            break
        if not guard.step():
            print(f"[TopologyGuard] {guard.faults[-1]['reason']} at t={guard.faults[-1]['time']:.3f}s -> restored")
        task.after_step()
    if task.report is None:
        if task.target_id is None:
            raise RuntimeError("Task timed out before planning")
        task.finish(timed_out=True)
    print_summary(task.report)
    print(f"\nTask report saved to {save_report(task.report, out_dir)}")
    return task.report
