"""Keyboard teleoperation for DogTaskSim simulation.

Wraps a pynput-based KeyboardController and adds grasp trigger + motion
detection used by SimulationScene.  Works alongside the GLFW PassiveViewer
window — pynput listens globally so keyboard commands work even when the
viewer window is focused.
"""

from __future__ import annotations

import math
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

from pynput import keyboard
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LegConfig:
    vel_x_max: float = 1.0
    vel_y_max: float = 0.8
    vel_yaw_max: float = 0.8
    accel: float = 0.5


@dataclass
class ArmConfig:
    x_range: list = field(default_factory=lambda: [0.4, 0.8])
    y_range: list = field(default_factory=lambda: [-0.35, 0.35])
    z_range: list = field(default_factory=lambda: [0.05, 1.0])
    mount_z_offset: float = 0.45
    accel: float = 0.5


@dataclass
class RPYConfig:
    roll_range: list = field(default_factory=lambda: [-30.0, 30.0])
    pitch_range: list = field(default_factory=lambda: [-45.0, 45.0])
    yaw_range: list = field(default_factory=lambda: [0.0, 0.0])
    step: float = 5.0


# Key sets for active-motion detection
LEG_KEYS = frozenset(["w", "s", "a", "d", "q", "e"])
ARM_KEYS = frozenset(["i", "k", "j", "l", "u", "o"])
ORI_KEYS = frozenset(["1", "2", "3", "4", "5", "6"])

HELP_TEXT = """
╔══════════════════════════════════╗
║  Keyboard Teleoperation          ║
╠══════════════════════════════════╣
║  [Leg]                           ║
║    W / S     Forward / Backward  ║
║    A / D     Left / Right        ║
║    Q / E     Turn Left / Right   ║
║                                  ║
║  [Arm End-Effector Position]     ║
║    I / K     X+ / X-             ║
║    J / L     Y+ / Y-             ║
║    U / O     Z+ / Z-             ║
║                                  ║
║  [End-Effector Orientation]      ║
║    1 / 2     Roll-  / Roll+      ║
║    3 / 4     Pitch- / Pitch+     ║
║    5 / 6     Yaw-   / Yaw+       ║
║                                  ║
║  [Other]                         ║
║    Space     Trigger grasp       ║
║    R         Stop all motion     ║
║    ESC       Exit                ║
╚══════════════════════════════════╝
"""


def print_state(velocity, pos, rpy_deg):
    vx, vy, vyaw = velocity
    x, y, z = pos[0], pos[1], pos[2]
    roll, pitch, yaw = rpy_deg

    lines = [
        "┌─────────── Command ───────────┐",
        f"│ vel : {vx:+.3f}  {vy:+.3f}  {vyaw:+.3f}     │",
        f"│ pos : {x:.3f}  {y:+.3f}  {z:.3f}     │",
        f"│ rpy : {roll:+.1f}°  {pitch:+.1f}°  {yaw:+.1f}°     │",
        "└─────────────────────────────────┘",
    ]

    sys.stdout.write("\033[2J\033[H")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


class KeyboardTeleop:
    """Keyboard teleoperation controller for SimulationScene.

    Call ``get_command()`` each frame to retrieve the latest command::

        {
            "velocity": [vx, vy, vyaw],
            "pos":      [x, y, z, qw, qx, qy, qz],
        }

    ``has_motion()`` returns True when any key is actively held.
    ``consume_grasp()`` returns True exactly once per Space press.
    """

    def __init__(
        self,
        leg_cfg: LegConfig | None = None,
        arm_cfg: ArmConfig | None = None,
        rpy_cfg: RPYConfig | None = None,
    ):
        self.leg_cfg = leg_cfg or LegConfig()
        self.arm_cfg = arm_cfg or ArmConfig()
        self.rpy_cfg = rpy_cfg or RPYConfig()

        self._velocity: list[float] = [0.0, 0.0, 0.0]
        self._pos: list[float] = [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
        self._rpy_deg: list[float] = [0.0, 0.0, 0.0]

        self._pressed_keys: set[str] = set()
        self._lock = threading.Lock()
        self._last_time: float = time.time()
        self.running: bool = True

        # One-shot triggers
        self._grasp_triggered: bool = False

        self._display_thread: Optional[threading.Thread] = None

        self._start_listener()
        print(HELP_TEXT)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_command(self) -> dict:
        self._update()
        with self._lock:
            return {
                "velocity": self._velocity.copy(),
                "pos": self._pos.copy(),
            }

    def has_motion(self) -> bool:
        """True when any key that moves the robot is held."""
        with self._lock:
            keys = frozenset(self._pressed_keys)
        return bool(keys & (LEG_KEYS | ARM_KEYS | ORI_KEYS))

    def consume_grasp(self) -> bool:
        """Return True exactly once per Space press, then reset."""
        with self._lock:
            if self._grasp_triggered:
                self._grasp_triggered = False
                return True
        return False

    def start_display(self, fps: float = 10.0) -> None:
        if self._display_thread is not None:
            return
        self._display_thread = threading.Thread(
            target=self._display_loop, args=(fps,), daemon=True
        )
        self._display_thread.start()

    def stop_motion(self) -> None:
        with self._lock:
            self._velocity = [0.0, 0.0, 0.0]
            self._pos = [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
            self._rpy_deg = [0.0, 0.0, 0.0]
            self._grasp_triggered = False

    def close(self) -> None:
        self.running = False
        if hasattr(self, "_listener") and self._listener.is_alive():
            self._listener.stop()

    # ------------------------------------------------------------------
    # Internal: pynput listener
    # ------------------------------------------------------------------

    def _start_listener(self) -> None:
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        try:
            self._listener.start()
        except Exception as exc:
            print(f"[KeyboardTeleop] Listener start failed: {exc}")

    def _on_press(self, key) -> None:
        try:
            char = key.char and key.char.lower()
            if char:
                with self._lock:
                    self._pressed_keys.add(char)
                return
        except AttributeError:
            pass
        # Special keys
        if key == keyboard.Key.space:
            with self._lock:
                self._grasp_triggered = True

    def _on_release(self, key) -> None:
        try:
            char = key.char and key.char.lower()
            if char:
                with self._lock:
                    self._pressed_keys.discard(char)
            elif key == keyboard.Key.esc:
                self.running = False
                return False
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # Internal: update
    # ------------------------------------------------------------------

    def _update(self) -> None:
        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        with self._lock:
            keys = frozenset(self._pressed_keys)

        self._update_leg(keys, dt)
        self._update_arm_pos(keys, dt)
        self._update_discrete(keys)
        self._update_arm_quat()

    def _update_leg(self, keys: frozenset, dt: float) -> None:
        inc = self.leg_cfg.accel * dt
        m = self.leg_cfg

        with self._lock:
            vx, vy, vyaw = self._velocity

        vx = self._ramp(vx, "w" in keys, "s" in keys, inc, m.vel_x_max)
        vy = self._ramp(vy, "a" in keys, "d" in keys, inc, m.vel_y_max)
        vyaw = self._ramp(vyaw, "q" in keys, "e" in keys, inc, m.vel_yaw_max)

        with self._lock:
            self._velocity = [vx, vy, vyaw]

    def _update_arm_pos(self, keys: frozenset, dt: float) -> None:
        inc = self.arm_cfg.accel * dt
        a = self.arm_cfg

        with self._lock:
            x, y, z = self._pos[0], self._pos[1], self._pos[2]

        x = self._ramp(x, "i" in keys, "k" in keys, inc, a.x_range[1], a.x_range[0])
        y = self._ramp(y, "j" in keys, "l" in keys, inc, a.y_range[1], a.y_range[0])
        z = self._ramp(z, "u" in keys, "o" in keys, inc, a.z_range[1], a.z_range[0])

        with self._lock:
            self._pos[0], self._pos[1], self._pos[2] = x, y, z

    def _update_discrete(self, keys: frozenset) -> None:
        with self._lock:
            r_deg, p_deg, y_deg = self._rpy_deg[:]
        rc, pc, yc = self.rpy_cfg, self.rpy_cfg, self.rpy_cfg
        step = self.rpy_cfg.step

        pairs = [
            ("1", lambda: max(rc.roll_range[0], r_deg - step), lambda v: self._set_rpy(roll=v)),
            ("2", lambda: min(rc.roll_range[1], r_deg + step), lambda v: self._set_rpy(roll=v)),
            ("3", lambda: max(pc.pitch_range[0], p_deg - step), lambda v: self._set_rpy(pitch=v)),
            ("4", lambda: min(pc.pitch_range[1], p_deg + step), lambda v: self._set_rpy(pitch=v)),
            ("5", lambda: max(yc.yaw_range[0], y_deg - step), lambda v: self._set_rpy(yaw=v)),
            ("6", lambda: min(yc.yaw_range[1], y_deg + step), lambda v: self._set_rpy(yaw=v)),
        ]

        for char, compute_fn, apply_fn in pairs:
            if char in keys:
                apply_fn(compute_fn())

        if "r" in keys:
            self.stop_motion()

    def _update_arm_quat(self) -> None:
        with self._lock:
            r_deg, p_deg, y_deg = self._rpy_deg[:]
            x, y_pos, z = self._pos[0], self._pos[1], self._pos[2]

        r = math.radians(r_deg)
        p = math.radians(p_deg)
        y = math.radians(y_deg)

        dz = z - self.arm_cfg.mount_z_offset
        dist_xy = math.hypot(x, y_pos)
        p -= math.atan2(dz, dist_xy)
        y += math.atan2(y_pos, x)

        xyzw = Rotation.from_euler("xyz", [r, p, y]).as_quat()
        qw, qx, qy, qz = xyzw[3], xyzw[0], xyzw[1], xyzw[2]

        with self._lock:
            self._pos[3:] = [qw, qx, qy, qz]

    def _display_loop(self, fps: float) -> None:
        interval = 1.0 / max(fps, 1.0)
        while self.running:
            with self._lock:
                vel = self._velocity[:]
                pos = self._pos[:]
                rpy_deg = self._rpy_deg[:]
            print_state(vel, pos, rpy_deg)
            time.sleep(interval)

    @staticmethod
    def _ramp(val, pos_key, neg_key, inc, upper, lower=None):
        if lower is None:
            lower = -upper
        if pos_key and not neg_key:
            return min(upper, val + inc)
        if neg_key and not pos_key:
            return max(lower, val - inc)
        return val

    def _set_rpy(self, roll=None, pitch=None, yaw=None):
        with self._lock:
            if roll is not None:
                self._rpy_deg[0] = roll
            if pitch is not None:
                self._rpy_deg[1] = pitch
            if yaw is not None:
                self._rpy_deg[2] = yaw
