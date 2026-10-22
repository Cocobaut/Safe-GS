"""Module 4 simulation - Collision verifier: contact extraction and collision-accuracy metrics.

Per step it records every contact (bodies, point, normal, penetration depth, contact force).
Across a run it measures:
  * interpenetration depth (max / mean / p95),
  * contacts of the probe with the target ROI,
  * error collisions against the M_High reference run:
      FP ("ghost" collision)  - the probe touches a body the reference run does not touch
                                at that step (e.g. a QEM-inflated surface),
      FN (missed collision)   - the reference touches it but this run does not
                                (e.g. a hole/handle closed or a surface lost by QEM),
  * Error_ROI - Chamfer distance between the probe-vs-ROI contact points of this run and of
                the reference run; compared against tau_ROI.
"""

import numpy as np
import mujoco
from scipy.spatial import cKDTree

PROBE_BODY = "probe"


def body_name(model, body_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or f"body_{body_id}"


def collect_contacts(model, data, target_body=None):
    """All active contacts of the current step as dicts."""
    contacts = []
    force = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        b1 = body_name(model, model.geom_bodyid[con.geom[0]])
        b2 = body_name(model, model.geom_bodyid[con.geom[1]])
        mujoco.mj_contactForce(model, data, i, force)
        contacts.append({
            "body1": b1,
            "body2": b2,
            "pos": np.array(con.pos),
            "normal": np.array(con.frame[:3]),
            "depth": max(0.0, -float(con.dist)),
            "normal_force": float(force[0]),
            "in_roi": target_body is not None and target_body in (b1, b2),
        })
    return contacts


class ContactRecorder:
    """Accumulates per-step contact information for one simulation run."""

    def __init__(self, model, target_body=None, probe_body=PROBE_BODY):
        self.model = model
        self.target_body = target_body
        self.probe_body = probe_body
        self.probe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, probe_body)
        self.max_depth_per_step = []
        self.ncon_per_step = []
        self.probe_touch = []       # per step: set of bodies the probe touches
        self.roi_points = []        # probe-vs-target contact points (whole run)
        self.roi_forces = []
        self.roi_steps = 0

    def record(self, data):
        """Record the current step. Contact forces are only evaluated for probe contacts,
        since this runs every step."""
        model = self.model
        probe_id = self.probe_id
        force = np.zeros(6)
        max_depth = 0.0
        touched = set()
        roi_contact = False
        for i in range(data.ncon):
            con = data.contact[i]
            max_depth = max(max_depth, -float(con.dist))
            b1 = int(model.geom_bodyid[con.geom[0]])
            b2 = int(model.geom_bodyid[con.geom[1]])
            if probe_id not in (b1, b2):
                continue
            other = body_name(model, b2 if b1 == probe_id else b1)
            touched.add(other)
            if other == self.target_body:
                roi_contact = True
                mujoco.mj_contactForce(model, data, i, force)
                self.roi_points.append(np.array(con.pos))
                self.roi_forces.append(float(force[0]))
        self.ncon_per_step.append(int(data.ncon))
        self.max_depth_per_step.append(max_depth)
        self.probe_touch.append(touched)
        self.roi_steps += roi_contact

    def summary(self):
        depths = np.asarray(self.max_depth_per_step) if self.max_depth_per_step else np.zeros(1)
        ncon = np.asarray(self.ncon_per_step) if self.ncon_per_step else np.zeros(1)
        return {
            "interpenetration_max_m": float(depths.max()),
            "interpenetration_mean_m": float(depths.mean()),
            "interpenetration_p95_m": float(np.percentile(depths, 95)),
            "contacts_mean": float(ncon.mean()),
            "contacts_std": float(ncon.std()),
            "roi_contact_steps": int(self.roi_steps),
            "roi_contact_points": len(self.roi_points),
            "roi_mean_normal_force_N": float(np.mean(self.roi_forces)) if self.roi_forces else 0.0,
        }


def chamfer(points_a, points_b):
    """Symmetric mean nearest-neighbour distance between two point sets (None if one is empty)."""
    if len(points_a) == 0 or len(points_b) == 0:
        return None
    a, b = np.asarray(points_a), np.asarray(points_b)
    return float((cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()) / 2.0)


def compare_to_reference(recorder, reference, tau_roi):
    """Error collisions and ROI contact error of `recorder` against the M_High `reference`."""
    steps = min(len(recorder.probe_touch), len(reference.probe_touch))
    fp = fn = ref_total = 0
    fp_bodies, fn_bodies = {}, {}
    for s in range(steps):
        run, ref = recorder.probe_touch[s], reference.probe_touch[s]
        ref_total += len(ref)
        for b in run - ref:
            fp += 1
            fp_bodies[b] = fp_bodies.get(b, 0) + 1
        for b in ref - run:
            fn += 1
            fn_bodies[b] = fn_bodies.get(b, 0) + 1
    error_roi = chamfer(recorder.roi_points, reference.roi_points)
    return {
        "false_positive_contacts": fp,
        "false_negative_contacts": fn,
        "false_positive_bodies": fp_bodies,
        "false_negative_bodies": fn_bodies,
        "reference_contacts": ref_total,
        "collision_accuracy": 1.0 - (fp + fn) / max(ref_total + fp, 1),
        "error_roi_m": error_roi,
        "tau_roi_m": float(tau_roi),
        "error_roi_within_tau": error_roi is not None and error_roi <= tau_roi,
    }
