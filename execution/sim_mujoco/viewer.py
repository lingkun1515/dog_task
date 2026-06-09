"""Passive GLFW viewer window for MuJoCo simulation visualisation."""

from __future__ import annotations

import os

import mujoco
from mujoco import glfw as _glfw


class PassiveViewer:
    """An interactive MuJoCo visualisation window driven by ``sync()``.

    Mouse controls:
      - Left drag: orbit
      - Right drag: pan
      - Scroll: zoom
      - R: reset camera
      - Esc: close window
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        width: int = 1200,
        height: int = 900,
        title: str = "DogTask Sim",
    ):
        self._window: int | None = None
        self._context: mujoco.MjrContext | None = None
        self._scene: mujoco.MjvScene | None = None
        self._cam: mujoco.MjvCamera | None = None
        self._opt: mujoco.MjvOption | None = None
        self._pert: mujoco.MjvPerturb | None = None
        self._width = width
        self._height = height

        self._init_glfw(model, title)

    # ------------------------------------------------------------------
    def _init_glfw(self, model: mujoco.MjModel, title: str) -> None:
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            raise RuntimeError("No display available (set DISPLAY or WAYLAND_DISPLAY)")

        if not _glfw.glfw.init():
            raise RuntimeError("GLFW initialisation failed")

        _glfw.glfw.window_hint(_glfw.glfw.DOUBLEBUFFER, 1)
        _glfw.glfw.window_hint(_glfw.glfw.SAMPLES, 4)
        _glfw.glfw.window_hint(_glfw.glfw.RESIZABLE, 1)

        window = _glfw.glfw.create_window(self._width, self._height, title, None, None)
        if window is None:
            _glfw.glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        _glfw.glfw.make_context_current(window)
        _glfw.glfw.swap_interval(1)

        self._window = window
        self._model = model
        self._context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self._scene = mujoco.MjvScene(model, maxgeom=10000)
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._cam.trackbodyid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        self._cam.distance = 4.0
        self._cam.elevation = -30
        self._cam.azimuth = 90
        self._opt = mujoco.MjvOption()
        self._pert = mujoco.MjvPerturb()

        _glfw.glfw.set_scroll_callback(window, self._scroll_callback)
        _glfw.glfw.set_cursor_pos_callback(window, self._mouse_move_callback)
        _glfw.glfw.set_mouse_button_callback(window, self._mouse_button_callback)
        _glfw.glfw.set_key_callback(window, self._key_callback)

        self._button_left = False
        self._button_middle = False
        self._button_right = False
        self._last_x = 0.0
        self._last_y = 0.0

    # ------------------------------------------------------------------
    # GLFW callbacks
    # ------------------------------------------------------------------
    def _scroll_callback(self, window, xoffset: float, yoffset: float) -> None:
        if self._cam is not None:
            mujoco.mjv_moveCamera(
                self._model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, yoffset * 0.05, self._scene, self._cam
            )

    def _mouse_button_callback(self, window, button: int, action: int, mods: int) -> None:
        pressed = action == _glfw.glfw.PRESS
        if button == _glfw.glfw.MOUSE_BUTTON_LEFT:
            self._button_left = pressed
        elif button == _glfw.glfw.MOUSE_BUTTON_MIDDLE:
            self._button_middle = pressed
        elif button == _glfw.glfw.MOUSE_BUTTON_RIGHT:
            self._button_right = pressed
        if pressed:
            xpos, ypos = _glfw.glfw.get_cursor_pos(window)
            self._last_x = xpos
            self._last_y = ypos

    def _mouse_move_callback(self, window, xpos: float, ypos: float) -> None:
        if self._cam is None or self._scene is None:
            return

        dx = xpos - self._last_x
        dy = ypos - self._last_y
        self._last_x = xpos
        self._last_y = ypos

        if self._button_left:
            mujoco.mjv_moveCamera(
                self._model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, dx * 0.005, dy * 0.005, self._scene, self._cam
            )
        elif self._button_middle:
            mujoco.mjv_moveCamera(
                self._model, mujoco.mjtMouse.mjMOUSE_MOVE_V, dx * 0.01, dy * 0.01, self._scene, self._cam
            )
        elif self._button_right:
            mujoco.mjv_moveCamera(
                self._model, mujoco.mjtMouse.mjMOUSE_ROTATE_H, dx * 0.005, dy * 0.005, self._scene, self._cam
            )

    def _key_callback(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        if action != _glfw.glfw.PRESS:
            return
        if key == _glfw.glfw.KEY_ESCAPE:
            _glfw.glfw.set_window_should_close(window, True)
        elif key == _glfw.glfw.KEY_R:
            if self._cam is not None:
                self._cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self._cam.distance = 4.0
                self._cam.elevation = -30
                self._cam.azimuth = 90

    # ------------------------------------------------------------------
    def sync(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """Render current state to the window.

        Returns True if the window is still open.
        """
        if self._window is None:
            return False

        if _glfw.glfw.window_should_close(self._window):
            return False

        viewport = mujoco.MjrRect(0, 0, self._width, self._height)
        mujoco.mjv_updateScene(
            model, data, self._opt, self._pert, self._cam,
            mujoco.mjtCatBit.mjCAT_ALL, self._scene,
        )
        mujoco.mjr_render(viewport, self._scene, self._context)
        _glfw.glfw.swap_buffers(self._window)
        _glfw.glfw.poll_events()

        return True

    # ------------------------------------------------------------------
    def should_close(self) -> bool:
        if self._window is None:
            return True
        return bool(_glfw.glfw.window_should_close(self._window))

    def close(self) -> None:
        if self._window is not None:
            _glfw.glfw.destroy_window(self._window)
            self._window = None
