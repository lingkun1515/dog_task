"""SceneBuilder: 组装 Go2 + D1 + Camera + 环境的 MuJoCo 场景.

使用 MjSpec API 的 attach() 方法将 D1 臂挂载到 Go2 底盘上。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ...config import project_path

logger = logging.getLogger(__name__)

GO2_LEG_JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

GO2_ACTUATORS = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

D1_ARM_JOINTS = ["arm_Joint1", "arm_Joint2", "arm_Joint3", "arm_Joint4", "arm_Joint5", "arm_Joint6"]
D1_GRIPPER_JOINTS = ["arm_Joint_L", "arm_Joint_R"]
D1_ALL_JOINTS = D1_ARM_JOINTS + D1_GRIPPER_JOINTS

GO2_HOME_QPOS = [0, 0.9, -1.8] * 4


class SceneBuilder:
    """组装 Go2+D1+Camera+Environment 的完整 MuJoCo 场景."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)

    def build(self) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """构建并编译统一场景，返回 (model, data)."""
        go2_xml_path = project_path(self._config["go2_xml"])
        d1_urdf_path = project_path(self._config["d1_urdf"])

        arm_mount_pos = self._config.get("arm_mount_pos", [0.0, 0.0, 0.10])
        arm_mount_quat = self._config.get("arm_mount_quat", [1, 0, 0, 0])
        camera_pos = self._config.get("camera_pos", [0.30, 0.0, 0.05])
        camera_fovy = self._config.get("camera_fovy", 58)
        camera_xyaxes = self._config.get("camera_xyaxes", "0 -1 0 0.0872 0 0.9962")
        camera_resolution = self._config.get("camera_resolution", [848, 480])
        env_config = self._config.get("environment", {})
        viewer = self._config.get("viewer", False)

        go2_spec = mujoco.MjSpec.from_file(str(go2_xml_path))
        d1_spec = mujoco.MjSpec.from_file(str(d1_urdf_path))

        base_link = go2_spec.worldbody.first_body()

        mount_site = base_link.add_site()
        mount_site.name = "arm_mount_site"
        mount_site.pos = arm_mount_pos
        mount_site.quat = arm_mount_quat

        go2_spec.attach(d1_spec, prefix="arm_", site=mount_site)

        cam = base_link.add_camera()
        cam.name = "d455_sim"
        cam.pos = camera_pos
        cam.fovy = camera_fovy
        cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        camera_quat = self._config.get("camera_quat", None)
        if camera_quat is not None:
            cam.quat = camera_quat
        else:
            cam.quat = self._xyaxes_to_quat(camera_xyaxes)

        ground = go2_spec.worldbody.add_geom()
        ground.name = "ground"
        ground.type = mujoco.mjtGeom.mjGEOM_PLANE
        ground_size = env_config.get("ground_size", [10, 10])
        ground.size = [ground_size[0], ground_size[1], 0.1]
        ground.rgba = [0.3, 0.4, 0.3, 1.0]

        light = go2_spec.worldbody.add_light()
        light.name = "overhead"
        light.pos = [0, 0, 5]
        light.dir = [0, 0, -1]
        light.diffuse = [0.8, 0.8, 0.8]
        light.castshadow = True

        for obj_cfg in env_config.get("objects", []):
            self._add_object(go2_spec, obj_cfg)

        offwidth = max(camera_resolution[0], 1280)
        offheight = max(camera_resolution[1], 720)
        go2_spec.visual.global_.offwidth = offwidth
        go2_spec.visual.global_.offheight = offheight

        model = go2_spec.compile()
        data = mujoco.MjData(model)

        keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if keyframe_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
        for name in D1_ALL_JOINTS:
            try:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                adr = model.jnt_qposadr[jid]
                data.qpos[adr] = 0.0
            except Exception:
                pass
        mujoco.mj_forward(model, data)

        logger.info(
            "Scene built: nq=%d, nv=%d, nu=%d, nbody=%d, ncam=%d",
            model.nq, model.nv, model.nu, model.nbody, model.ncam,
        )
        return model, data

    @staticmethod
    def _xyaxes_to_quat(xyaxes) -> List[float]:
        """将 MuJoCo xyaxes (6 floats) 转为 [w,x,y,z] 四元数."""
        if isinstance(xyaxes, str):
            vals = [float(v) for v in xyaxes.split()]
        else:
            vals = [float(v) for v in xyaxes]
        x_axis = np.array(vals[0:3])
        y_axis = np.array(vals[3:6])
        z_axis = np.cross(x_axis, y_axis)
        R = np.column_stack([x_axis, y_axis, z_axis])
        q_xyzw = Rotation.from_matrix(R).as_quat()
        return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]

    def _add_object(self, spec: mujoco.MjSpec, obj_cfg: Dict[str, Any]) -> None:
        """向场景添加一个环境物体."""
        obj_type = obj_cfg.get("type", "sphere")
        name = obj_cfg.get("name", "object")
        pos = obj_cfg.get("pos", [0, 0, 0])
        size = obj_cfg.get("size", 0.025)
        rgba = obj_cfg.get("rgba", [0.95, 0.25, 0.15, 1.0])
        is_mocap = obj_cfg.get("mocap", False)

        type_map = {
            "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
            "box": mujoco.mjtGeom.mjGEOM_BOX,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        }
        geom_type = type_map.get(obj_type, mujoco.mjtGeom.mjGEOM_SPHERE)

        if is_mocap:
            body = spec.worldbody.add_body()
            body.name = name
            body.pos = pos
            body.mocap = True
            geom = body.add_geom()
            geom.name = f"{name}_geom"
            geom.type = geom_type
            if isinstance(size, (list, tuple)):
                geom.size = size
            else:
                geom.size = [size, 0, 0]
            geom.rgba = rgba
            geom.contype = 0
            geom.conaffinity = 0
        else:
            geom = spec.worldbody.add_geom()
            geom.name = name
            geom.type = geom_type
            geom.pos = pos
            if isinstance(size, (list, tuple)):
                geom.size = size
            else:
                geom.size = [size, 0, 0]
            geom.rgba = rgba
