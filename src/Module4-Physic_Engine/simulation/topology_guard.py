"""Module 4 simulation - Topology Guard: asset checks before loading + a 5-state run-time safety machine.

    IDLE -> VALIDATING -> SETTLING -> RUNNING
                 |            |          |
                 +--------> FAULT <------+   (restore last safe snapshot, then SETTLING again)

* VALIDATING: every mesh the scene references is checked (non-empty, finite, valid indices,
  few degenerate triangles; collision hulls need >= 4 vertices). Invalid collision hulls are
  dropped instead of crashing the MuJoCo compiler; an invalid visual mesh is a hard error.
* SETTLING / RUNNING: each step is checked for NaN/Inf state, MuJoCo's BADQACC warning and
  runaway speeds ("simulation explosion"). A fault restores the last healthy snapshot with
  zero velocity, so one bad contact cannot blow up the whole scene; faults are counted and
  reported as the explosion rate by benchmarks.py.
* Sleeping (`sleep=True`, used by the viewer): every dynamic body starts held at its scanned
  pose and only wakes when it is disturbed - dragged in the viewer (xfrc_applied) or touched
  by an awake body / the probe. A freshly woken body has its speed clamped for
  WAKE_SETTLE_TIME so overlaps inherited from the scan separate gently instead of popping.
"""

import os
import sys

import numpy as np
import mujoco
import open3d as o3d

# src/Module4-Physic_Engine/ (holds preprocessing/ and simulation/)
ENGINE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)


from preprocessing.mesh_validation import check_mesh_structure

IDLE, VALIDATING, SETTLING, RUNNING, FAULT = "IDLE", "VALIDATING", "SETTLING", "RUNNING", "FAULT"
STATES = (IDLE, VALIDATING, SETTLING, RUNNING, FAULT)

SETTLE_TIME = 0.5       # s spent in SETTLING before the task (probe) may act
MAX_LINEAR_SPEED = 20.0  # m/s; anything faster in an indoor scene is an explosion
MAX_ANGULAR_SPEED = 200.0  # rad/s
WAKE_SETTLE_TIME = 0.5   # s after waking during which speed is clamped
WAKE_MAX_LINEAR = 0.5    # m/s
WAKE_MAX_ANGULAR = 5.0   # rad/s


def validate_mesh_file(path, collision=False):
    """Return (ok, status) for one mesh asset."""
    if not os.path.exists(path):
        return False, "MISSING_FILE"
    mesh = o3d.io.read_triangle_mesh(path)
    ok, status = check_mesh_structure(mesh)
    if ok and collision and len(np.unique(np.asarray(mesh.vertices), axis=0)) < 4:
        return False, "HULL_UNDER_4_VERTICES"
    return ok, status


def validate_assets(visual_paths, collision_paths):
    """Check every scene asset. Returns (invalid_visual, invalid_collision) as {path: status}."""
    bad_visual = {}
    for path in visual_paths:
        ok, status = validate_mesh_file(path)
        if not ok:
            bad_visual[path] = status
    bad_collision = {}
    for path in collision_paths:
        ok, status = validate_mesh_file(path, collision=True)
        if not ok:
            bad_collision[path] = status
    return bad_visual, bad_collision


class TopologyGuard:
    """Wraps mj_step with health checks and rollback. Use `step()` instead of `mujoco.mj_step`."""

    def __init__(self, model, data, settle_time=SETTLE_TIME,
                 max_linear_speed=MAX_LINEAR_SPEED, max_angular_speed=MAX_ANGULAR_SPEED, sleep=False):
        self.model = model
        self.data = data
        self.settle_time = settle_time
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.state = IDLE
        self.faults = []
        self.steps = 0
        self._settle_start = 0.0
        self._free_dofs = [
            int(model.jnt_dofadr[j]) for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        self._snapshot = None
        self._badqacc_seen = 0

        # Sleeping: body id -> (qpos adr, dof adr) of its free joint
        self._free_bodies = {
            int(model.jnt_bodyid[j]): (int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))
            for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        }
        self.sleep_enabled = sleep
        self.asleep = set()
        self.woken = {}         # body id -> time it woke
        self.unclamped = set()  # bodies exempt from the wake speed clamp (task actor / target)
        self._rest_qpos = {}
        self._last_time = 0.0
        # Colliding geoms per free body; mocap (probe) geoms must also reach sleeping bodies.
        self._body_geoms = {
            b: [g for g in range(model.ngeom)
                if model.geom_bodyid[g] == b and (model.geom_contype[g] or model.geom_conaffinity[g])]
            for b in self._free_bodies
        }
        if sleep:
            for g in range(model.ngeom):
                if model.body_mocapid[model.geom_bodyid[g]] >= 0 and model.geom_contype[g]:
                    model.geom_contype[g] |= self.AWAKE_CONTYPE

    # -- sleeping -------------------------------------------------------------------------
    # Collision bits: static geoms (1, 1); awake bodies / probe (3, 1); sleeping bodies (0, 2).
    # Sleeping bodies then only collide with awake bodies and the probe - no solver work is
    # spent on overlaps inherited from the scan while everything rests.
    AWAKE_CONTYPE, AWAKE_CONAFFINITY = 3, 1
    SLEEP_CONTYPE, SLEEP_CONAFFINITY = 0, 2

    def _set_collision(self, body, asleep):
        ct, ca = ((self.SLEEP_CONTYPE, self.SLEEP_CONAFFINITY) if asleep
                  else (self.AWAKE_CONTYPE, self.AWAKE_CONAFFINITY))
        for g in self._body_geoms[body]:
            self.model.geom_contype[g] = ct
            self.model.geom_conaffinity[g] = ca

    def _put_all_to_sleep(self):
        self.asleep = set(self._free_bodies) if self.sleep_enabled else set()
        self.woken = {}
        self.unclamped = set()
        self._rest_qpos = {b: self.data.qpos[qa:qa + 7].copy() for b, (qa, _) in self._free_bodies.items()}
        if self.sleep_enabled:
            for b in self._free_bodies:
                self._set_collision(b, asleep=True)

    def wake(self, body):
        if body in self.asleep:
            self.asleep.discard(body)
            self.woken[body] = float(self.data.time)
            self._set_collision(body, asleep=False)

    def _update_sleep(self):
        m, d = self.model, self.data
        if d.time < self._last_time:          # viewer reset (Backspace): back to sleep
            self._put_all_to_sleep()
        self._last_time = float(d.time)
        if self.asleep:
            for b in list(self.asleep):
                if np.any(d.xfrc_applied[b] != 0.0):   # dragged with the mouse
                    self.wake(b)
            # Any active contact counts, also one still inside the contact margin (dist >= 0):
            # it already pushes, and a held sleeper would act as an immovable wall. Sleepers do
            # not collide with each other, so this only reacts to awake bodies and the probe.
            for i in range(d.ncon):
                con = d.contact[i]
                b1, b2 = int(m.geom_bodyid[con.geom[0]]), int(m.geom_bodyid[con.geom[1]])
                for sleeper, other in ((b1, b2), (b2, b1)):
                    if sleeper in self.asleep and (
                        (other in self._free_bodies and other not in self.asleep) or m.body_mocapid[other] >= 0
                    ):
                        self.wake(sleeper)
            for b in self.asleep:
                qa, da = self._free_bodies[b]
                d.qpos[qa:qa + 7] = self._rest_qpos[b]
                d.qvel[da:da + 6] = 0.0
        for b, t in list(self.woken.items()):
            if d.time - t > WAKE_SETTLE_TIME or b in self.unclamped:
                continue
            _, da = self._free_bodies[b]
            for sl, cap in ((slice(da, da + 3), WAKE_MAX_LINEAR), (slice(da + 3, da + 6), WAKE_MAX_ANGULAR)):
                speed = np.linalg.norm(d.qvel[sl])
                if speed > cap:
                    d.qvel[sl] *= cap / speed


    # -- state transitions -------------------------------------------------------------
    def validate(self, visual_paths=(), collision_paths=()):
        """IDLE -> VALIDATING -> SETTLING (or FAULT when a visual mesh is unusable)."""
        self.state = VALIDATING
        bad_visual, bad_collision = validate_assets(visual_paths, collision_paths)
        self.validation = {"invalid_visual": bad_visual, "invalid_collision": bad_collision}
        if bad_visual:
            self.state = FAULT
            return False
        self._enter_settling()
        return True

    def start(self):
        """Skip asset validation (already done when the scene was built)."""
        self.validation = {"invalid_visual": {}, "invalid_collision": {}}
        self._enter_settling()

    def _enter_settling(self):
        mujoco.mj_forward(self.model, self.data)
        if not self._rest_qpos:
            self._put_all_to_sleep()
            self._last_time = float(self.data.time)
        self._take_snapshot()
        self._settle_start = float(self.data.time)
        self.state = SETTLING

    @property
    def running(self):
        return self.state == RUNNING

    # -- stepping ------------------------------------------------------------------------
    def step(self):
        if self.state in (IDLE, VALIDATING):
            raise RuntimeError(f"TopologyGuard.step() called in state {self.state}; call validate()/start() first")
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if self.sleep_enabled:
            self._update_sleep()
        reason = self._health_problem()
        if reason:
            self._fault(reason)
            return False
        self._take_snapshot()
        if self.state == SETTLING and self.data.time - self._settle_start >= self.settle_time:
            self.state = RUNNING
        return True

    def _health_problem(self):
        d = self.data
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)) and np.all(np.isfinite(d.qacc))):
            return "NAN_STATE"
        badqacc = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        if badqacc > self._badqacc_seen:
            self._badqacc_seen = badqacc
            return "BAD_QACC"
        for adr in self._free_dofs:
            if np.linalg.norm(d.qvel[adr:adr + 3]) > self.max_linear_speed:
                return "LINEAR_SPEED"
            if np.linalg.norm(d.qvel[adr + 3:adr + 6]) > self.max_angular_speed:
                return "ANGULAR_SPEED"
        return None

    def _take_snapshot(self):
        d = self.data
        self._snapshot = (d.time, d.qpos.copy(), d.mocap_pos.copy(), d.mocap_quat.copy())

    def _fault(self, reason):
        """RUNNING/SETTLING -> FAULT -> (restore) -> SETTLING."""
        self.state = FAULT
        self.faults.append({"time": float(self.data.time), "step": self.steps, "reason": reason})
        t, qpos, mocap_pos, mocap_quat = self._snapshot
        d = self.data
        d.qpos[:] = qpos
        d.qvel[:] = 0.0
        d.qacc_warmstart[:] = 0.0
        d.mocap_pos[:] = mocap_pos
        d.mocap_quat[:] = mocap_quat
        d.time = t
        self._last_time = float(t)
        self._enter_settling()

    def report(self):
        return {
            "final_state": self.state,
            "steps": self.steps,
            "faults": self.faults,
            "fault_count": len(self.faults),
            "explosion_rate": len(self.faults) / self.steps if self.steps else 0.0,
            "sleep": self.sleep_enabled,
            "woken_bodies": sorted(
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.woken
            ),
            **getattr(self, "validation", {}),
        }
