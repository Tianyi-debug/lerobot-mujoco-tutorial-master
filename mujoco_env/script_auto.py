# mujoco_env/script_auto.py
import numpy as np

class ScriptPolicy:
    """
    Auto pick-and-place scripted policy.
    Drop-in replacement for `teleop_robot()`:
        action, done = policy(PnPEnv)

    Output action (7,):
        [dx, dy, dz, droll, dpitch, dyaw, gripper_state]
    where gripper_state is a HOLD state (0=open, 1=close),
    consistent with SimpleEnv.teleop_robot() semantics. :contentReference[oaicite:1]{index=1}
    """

    def __init__(
        self,
        # ---------------- motion step limits (per call) ----------------
        max_dpos=0.007,          # match keyboard teleop step :contentReference[oaicite:2]{index=2}
        max_drot=0.0,            # keep orientation fixed by default
        # ---------------- stage switching thresholds ----------------
        xy_thresh=0.012,
        z_thresh=0.012,
        # ---------------- geometry / offsets ----------------
        pregrasp_hover=0.18,     # above mug before descend (>=0.12 recommended)
        grasp_z_offset=0.020,    # descend target = mug_z + grasp_z_offset
        safe_clearance=0.30,     # KEY: carry height = max(mug_z, plate_z) + safe_clearance
        place_hover=0.25,        # above plate before open (keep big to avoid collision)
        place_z_offset=0.08,     # optional: if you want descend closer before open (not used by default)
        retreat_clearance=0.30,  # retreat height relative to objects
        # ---------------- timing / holds ----------------
        close_hold_steps=25,
        open_hold_steps=18,
        settle_steps=10,
        # ---------------- safety ----------------
        max_steps_per_episode=2000,
        # ---------------- names (align to your XML) ----------------
        tcp_body="tcp_link",
        plate_body="body_obj_plate_11",
        default_mug_body="body_obj_mug_5",
        # ---------------- behavior toggles ----------------
        use_place_descend=False,  # if True: add a small descend before opening
    ):
        # store ALL params on self (no local var trap)
        self.max_dpos = float(max_dpos)
        self.max_drot = float(max_drot)

        self.xy_thresh = float(xy_thresh)
        self.z_thresh = float(z_thresh)

        self.pregrasp_hover = float(pregrasp_hover)
        self.grasp_z_offset = float(grasp_z_offset)
        self.safe_clearance = float(safe_clearance)
        self.place_hover = float(place_hover)
        self.place_z_offset = float(place_z_offset)
        self.retreat_clearance = float(retreat_clearance)

        self.close_hold_steps = int(close_hold_steps)
        self.open_hold_steps = int(open_hold_steps)
        self.settle_steps = int(settle_steps)

        self.max_steps_per_episode = int(max_steps_per_episode)

        self.tcp_body = str(tcp_body)
        self.plate_body = str(plate_body)
        self.default_mug_body = str(default_mug_body)

        self.use_place_descend = bool(use_place_descend)

        self.reset()

    # ---------------- public API ----------------
    def reset(self):
        self.stage = "pregrasp"
        self.gripper_state = 0.0  # 0=open, 1=close
        self._hold_counter = 0
        self._settle_counter = 0
        self._step_counter = 0

    def __call__(self, env):
        """
        env is your PnPEnv wrapper (SimpleEnv / SimpleEnv2 instance).
        Must provide: env.env.get_p_body(body_name)
        """
        self._step_counter += 1
        if self._step_counter >= self.max_steps_per_episode:
            # fail-safe: end episode if stuck
            return self._action_hold(gripper=0.0), True

        # --- resolve object names (SimpleEnv2 may have env.obj_target) ---
        mug_body = getattr(env, "obj_target", self.default_mug_body)
        plate_body = self.plate_body
        tcp_body = self.tcp_body

        # --- read poses from mujoco (ground truth) ---
        p_tcp = env.env.get_p_body(tcp_body)
        p_mug = env.env.get_p_body(mug_body)
        p_plate = env.env.get_p_body(plate_body)

        # --- compute safe heights dynamically (important) ---
        # objects z are around 0.82 in your reset sampling :contentReference[oaicite:3]{index=3}
        z_base = float(max(p_mug[2], p_plate[2]))
        z_carry = z_base + self.safe_clearance
        z_retreat = z_base + self.retreat_clearance

        # ---------------- state machine ----------------
        if self.stage == "pregrasp":
            # move above mug, open gripper
            target = np.array([p_mug[0], p_mug[1], p_mug[2] + self.pregrasp_hover], dtype=float)
            self.gripper_state = 0.0
            action = self._goto_xyz(p_tcp, target, gripper=self.gripper_state)

            if self._reached_xy(p_tcp, target) and self._reached_z(p_tcp, target):
                self.stage = "descend"
            return action, False

        if self.stage == "descend":
            # descend to grasp height, keep xy aligned
            target = np.array([p_mug[0], p_mug[1], p_mug[2] + self.grasp_z_offset], dtype=float)
            self.gripper_state = 0.0
            action = self._goto_xyz(p_tcp, target, gripper=self.gripper_state)

            if self._reached_xy(p_tcp, target) and self._reached_z(p_tcp, target):
                self.stage = "close_gripper"
                self._hold_counter = 0
            return action, False

        if self.stage == "close_gripper":
            # close and hold (let contact settle)
            self.gripper_state = 1.0
            action = self._action_hold(gripper=self.gripper_state)
            self._hold_counter += 1

            if self._hold_counter >= self.close_hold_steps:
                self.stage = "lift"
            return action, False

        if self.stage == "lift":
            # lift straight up to carry height (critical to avoid sweeping plate)
            target = np.array([p_tcp[0], p_tcp[1], z_carry], dtype=float)
            self.gripper_state = 1.0
            action = self._goto_xyz(p_tcp, target, gripper=self.gripper_state)

            if self._reached_z(p_tcp, target):
                self.stage = "move_to_target"
            return action, False

        if self.stage == "move_to_target":
            # move in XY to above plate, keep z at carry height (critical)
            target = np.array([p_plate[0], p_plate[1], z_carry], dtype=float)
            self.gripper_state = 1.0
            action = self._goto_xyz(p_tcp, target, gripper=self.gripper_state)

            if self._reached_xy(p_tcp, target) and self._reached_z(p_tcp, target):
                self.stage = "place"
            return action, False

        if self.stage == "place":
            # descend to a SAFE hover above plate (do NOT go too low)
            target = np.array([p_plate[0], p_plate[1], p_plate[2] + self.place_hover], dtype=float)
            self.gripper_state = 1.0
            action = self._goto_xyz(p_tcp, target, gripper=self.gripper_state)

            if self._reached_xy(p_tcp, target) and self._reached_z(p_tcp, target):
                if self.use_place_descend:
                    self.stage = "place_descend"
                else:
                    self.stage = "open_gripper"
                    self._hold_counter = 0
            return action, False

        if self.stage == "place_descend":
            # OPTIONAL: descend closer before opening
            target = np.array([p_plate[0], p_plate[1], p_plate[2] + self.place_z_offset], dtype=float)
            self.gripper_state = 1.0
            action = self._goto_xyz(p_tcp, target, gripper=self.gripper_state)

            if self._reached_xy(p_tcp, target) and self._reached_z(p_tcp, target):
                self.stage = "open_gripper"
                self._hold_counter = 0
            return action, False

        if self.stage == "open_gripper":
            # open and hold
            self.gripper_state = 0.0
            action = self._action_hold(gripper=self.gripper_state)
            self._hold_counter += 1

            if self._hold_counter >= self.open_hold_steps:
                self.stage = "retreat"
                self._settle_counter = 0
            return action, False

        if self.stage == "retreat":
            # retreat upward and finish episode
            target = np.array([p_tcp[0], p_tcp[1], z_retreat], dtype=float)
            self.gripper_state = 0.0
            action = self._goto_xyz(p_tcp, target, gripper=self.gripper_state)

            if self._reached_z(p_tcp, target):
                self._settle_counter += 1
                if self._settle_counter >= self.settle_steps:
                    return self._action_hold(gripper=0.0), True
            return action, False

        # fallback
        return self._action_hold(gripper=0.0), True

    # ---------------- helpers ----------------
    def _clip_vec(self, v, max_norm):
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return v
        if n > max_norm:
            return v * (max_norm / n)
        return v

    def _reached_xy(self, p, target):
        return float(np.linalg.norm(p[:2] - target[:2])) < self.xy_thresh

    def _reached_z(self, p, target):
        return abs(float(p[2] - target[2])) < self.z_thresh

    def _goto_xyz(self, p_tcp, target_xyz, gripper):
        dpos = target_xyz - p_tcp
        dpos = self._clip_vec(dpos, self.max_dpos)

        # keep orientation fixed unless you later enable rotation
        drot_rpy = np.zeros(3, dtype=float)
        action = np.concatenate([dpos, drot_rpy, np.array([gripper], dtype=float)], dtype=np.float32)
        return action

    def _action_hold(self, gripper):
        return np.array([0, 0, 0, 0, 0, 0, float(gripper)], dtype=np.float32)
