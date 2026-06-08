from __future__ import annotations

import logging
import socket
import struct
import time
from enum import IntEnum
from typing import NamedTuple

logger = logging.getLogger(__name__)

MSG_KEY_VALUE = 0x1008
MSG_HEADER_FORMAT = "!HHI"
MSG_HEADER_SIZE = struct.calcsize(MSG_HEADER_FORMAT)
MSG_KEY_VALUE_BYTES = struct.pack("!H", MSG_KEY_VALUE)
MAX_PAYLOAD_SIZE = 4 * 1024 * 1024
RECEIVE_CHUNK_SIZE = 4096
CTRL_INFO_SIZE = 32
CTRL_INFO_COMMAND_FORMAT = "=HHH"
COMMON_DATA_CTRL_MOTION_OFFSET = 20
COMMON_DATA_CTRL_MOTION_STATE_OFFSET = 24
COMMON_DATA_CTRL_MOTION_STATE_SIZE = struct.calcsize("=i")

GOATBOT_TCP_DATA_TYPE_COMMON = 0x00A1
GOATBOT_TCP_CTRL_TYPE_REMOTE_CTRL = 0x00B6


class RemoteKey(IntEnum):
    """遥控指令枚举，对应 C 端 KeyValue。"""

    RECHARGE = 57343
    STOP = 65535
    COVER = 65531
    DEPART = 32767


class MotionFsm(IntEnum):
    """机器人运动模式，对应 C 端 MotionFsm。"""

    UNKNOWN = -1
    NO_STAGE = 0
    STOP = 1
    STAGE_ARC = 2
    STAGE_AP2P = 3
    STAGE_FE = 4
    STAGE_LP2P = 5
    STAGE_NP2P = 6
    STAGE_TRAP = 7
    STAGE_REMOTE = 8
    STAGE_DEPARTURE = 9
    STAGE_EXCEPTION = 10
    STAGE_TRACKING = 11
    STAGE_FO = 12
    STAGE_RECHARGE = 13
    STAGE_RELOCATION = 14
    STAGE_FACTORY_TEST = 15


class ControlState(IntEnum):
    """机器人运动模式状态，对应 C 端 ControlState。"""

    UNKNOWN = -1
    IDLE = 0
    RUN = 1
    FAIL = 2
    FINISH = 3
    WAIT = 4
    TIME_OUT = 5
    OUT_MAP = 6


class MotionInfo(NamedTuple):
    ctrl_motion: MotionFsm
    ctrl_motion_state: ControlState


def build_remote_control_message(
    cmd_func: int | RemoteKey,
    cmd_v: int = 127,
    cmd_w: int = 127,
) -> bytes:
    """构造远程控制报文，其中 cmd_func 对应 CTRL_INFO.UINT16[2]。"""
    _validate_uint16("cmd_v", cmd_v)
    _validate_uint16("cmd_w", cmd_w)
    _validate_uint16("cmd_func", cmd_func)

    payload = bytearray(CTRL_INFO_SIZE)
    struct.pack_into(CTRL_INFO_COMMAND_FORMAT, payload, 0, cmd_v, cmd_w, cmd_func)
    header = struct.pack(
        MSG_HEADER_FORMAT,
        MSG_KEY_VALUE,
        GOATBOT_TCP_CTRL_TYPE_REMOTE_CTRL,
        len(payload),
    )
    return header + payload


def _validate_uint16(name: str, value: int) -> None:
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"{name} 必须在 0 到 65535 之间")


class SocketConnection:
    """与对端建立 TCP 连接，并收发原始字节数据。"""

    def __init__(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: socket.socket | None = None
        self._receive_buffer = bytearray()

    @property
    def is_connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> bool:
        """连接对端。重复调用时保持当前连接。"""
        if self.is_connected:
            return True

        try:
            self._socket = socket.create_connection(
                (self.host, self.port),
                timeout=self.timeout,
            )
        except OSError:
            logger.exception("连接对端失败: %s:%s", self.host, self.port)
            return False
        return True

    def send(self, data: bytes) -> None:
        """向对端发送完整的字节数据。"""
        connection = self._require_connection()
        connection.sendall(data)

    def send_remote_control(
        self,
        cmd_func: int | RemoteKey,
        cmd_v: int = 127,
        cmd_w: int = 127,
    ) -> None:
        """发送远程控制指令，其中 cmd_func 对应 CTRL_INFO.UINT16[2]。"""
        self.send(build_remote_control_message(cmd_func, cmd_v, cmd_w))

    def receive_motion_info(self) -> MotionInfo:
        """接收一帧 CommonData 报文并解析 ctrl_motion 与 ctrl_motion_state。"""
        payload = self._receive_common_data_message()
        return parse_motion_info(payload)

    def go_to_location(
        self,
        run_timeout: float = 10.0,
        finish_timeout: float = 300.0,
    ) -> bool:
        """启动覆盖模式，进入 RUN 后等待 FINISH。"""
        return self._run_remote_control(
            RemoteKey.COVER,
            MotionFsm.STAGE_LP2P,
            run_timeout,
            finish_timeout,
        )

    def go_docking(
        self,
        run_timeout: float = 10.0,
        finish_timeout: float = 300.0,
    ) -> bool:
        """启动回充模式，进入 RUN 后等待 FINISH。"""
        return self._run_remote_control(
            RemoteKey.RECHARGE,
            MotionFsm.STAGE_RECHARGE,
            run_timeout,
            finish_timeout,
        )

    def _run_remote_control(
        self,
        cmd_func: int | RemoteKey,
        ctrl_motion: MotionFsm,
        run_timeout: float,
        finish_timeout: float,
    ) -> bool:
        connection = self._require_connection()
        self.send_remote_control(cmd_func)

        run_deadline = time.monotonic() + run_timeout
        original_timeout = connection.gettimeout()
        try:
            while True:
                remaining = run_deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning("等待 RUN 超时 (run_timeout=%s)", run_timeout)
                    return False

                connection.settimeout(remaining)
                try:
                    motion = self.receive_motion_info()
                    logger.info(
                        "receive_motion_info: ctrl_motion=%s, ctrl_motion_state=%s",
                        motion.ctrl_motion,
                        motion.ctrl_motion_state,
                    )
                except TimeoutError:
                    logger.warning("等待 RUN 超时 (run_timeout=%s)", run_timeout)
                    return False

                if motion.ctrl_motion_state is ControlState.RUN:
                    break

            finish_deadline = time.monotonic() + finish_timeout
            while True:
                remaining = finish_deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "等待任务完成超时 (finish_timeout=%s)", finish_timeout
                    )
                    return False

                connection.settimeout(remaining)
                try:
                    motion = self.receive_motion_info()
                    logger.info(
                        "receive_motion_info: ctrl_motion=%s, ctrl_motion_state=%s",
                        motion.ctrl_motion,
                        motion.ctrl_motion_state,
                    )
                except TimeoutError:
                    logger.warning(
                        "等待任务完成超时 (finish_timeout=%s)", finish_timeout
                    )
                    return False

                if (
                    motion.ctrl_motion_state is ControlState.IDLE
                    and motion.ctrl_motion is MotionFsm.NO_STAGE
                ):
                    return True
                if motion.ctrl_motion is not ctrl_motion:
                    continue
                if motion.ctrl_motion_state in (
                    ControlState.FAIL,
                    # ControlState.TIME_OUT,
                    # ControlState.OUT_MAP,
                ):
                    return False
                if motion.ctrl_motion_state is ControlState.FINISH:
                    return True
        finally:
            connection.settimeout(original_timeout)

    def close(self) -> None:
        """关闭连接。"""
        if self._socket is None:
            return

        self._socket.close()
        self._socket = None
        self._receive_buffer.clear()

    def _require_connection(self) -> socket.socket:
        if self._socket is None:
            raise RuntimeError("Socket 尚未连接，请先调用 connect()")
        return self._socket

    def _receive_common_data_message(self) -> bytes:
        while True:
            self._align_to_header()
            self._fill_receive_buffer(MSG_HEADER_SIZE)
            _, message_type, payload_size = struct.unpack_from(
                MSG_HEADER_FORMAT,
                self._receive_buffer,
            )
            if message_type != GOATBOT_TCP_DATA_TYPE_COMMON:
                if not self._discard_buffered_message(message_type, payload_size):
                    del self._receive_buffer[0]
                continue
            if payload_size < COMMON_DATA_CTRL_MOTION_STATE_OFFSET + COMMON_DATA_CTRL_MOTION_STATE_SIZE:
                logger.error("丢弃异常消息头，payload 长度过小: %s", payload_size)
                del self._receive_buffer[0]
                continue
            if payload_size > MAX_PAYLOAD_SIZE:
                logger.warning("丢弃异常消息头，payload 长度过大: %s", payload_size)
                del self._receive_buffer[0]
                continue
            message_size = MSG_HEADER_SIZE + payload_size
            self._fill_receive_buffer(message_size)
            payload = bytes(self._receive_buffer[MSG_HEADER_SIZE:message_size])
            del self._receive_buffer[:message_size]
            return payload

    def _discard_buffered_message(self, message_type: int, payload_size: int) -> bool:
        """按报文头 Length 读取完整 payload 后丢弃。长度非法时返回 False。"""
        if payload_size > MAX_PAYLOAD_SIZE:
            logger.warning(
                "无法按帧丢弃消息，payload 长度过大: type=0x%04X, len=%s",
                message_type,
                payload_size,
            )
            return False

        message_size = MSG_HEADER_SIZE + payload_size
        self._fill_receive_buffer(message_size)
        del self._receive_buffer[:message_size]
        logger.debug(
            "丢弃非 CommonData 消息: type=0x%04X, payload_len=%s",
            message_type,
            payload_size,
        )
        return True

    def _align_to_header(self) -> None:
        while True:
            header_index = self._receive_buffer.find(MSG_KEY_VALUE_BYTES)
            if header_index >= 0:
                if header_index:
                    logger.warning("丢弃 Header 前的 %s 个无效字节", header_index)
                    del self._receive_buffer[:header_index]
                return

            keep_size = 1 if self._receive_buffer.endswith(MSG_KEY_VALUE_BYTES[:1]) else 0
            if len(self._receive_buffer) > keep_size:
                discarded_size = len(self._receive_buffer) - keep_size
                logger.warning("丢弃 %s 个无法对齐 Header 的字节", discarded_size)
                del self._receive_buffer[:discarded_size]
            self._receive_into_buffer()

    def _fill_receive_buffer(self, size: int) -> None:
        while len(self._receive_buffer) < size:
            self._receive_into_buffer()

    def _receive_into_buffer(self) -> None:
        connection = self._require_connection()
        chunk = connection.recv(RECEIVE_CHUNK_SIZE)
        if not chunk:
            raise ConnectionError("对端在完整报文接收完成前关闭了连接")  # TODO(sgk): 不要抛出异常
        self._receive_buffer.extend(chunk)

    def __enter__(self) -> SocketConnection:
        if not self.connect():
            raise ConnectionError(f"连接对端失败: {self.host}:{self.port}")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def parse_motion_info(payload: bytes) -> MotionInfo:
    """从 CommonData payload 中解析 ctrl_motion 与 ctrl_motion_state。"""
    required_size = (
        COMMON_DATA_CTRL_MOTION_STATE_OFFSET + COMMON_DATA_CTRL_MOTION_STATE_SIZE
    )
    if len(payload) < required_size:
        raise ValueError(f"CommonData payload 长度不足: {len(payload)} < {required_size}")

    ctrl_motion_value, state_value = struct.unpack_from(
        "=ii", payload, COMMON_DATA_CTRL_MOTION_OFFSET
    )
    try:
        ctrl_motion = MotionFsm(ctrl_motion_value)
    except ValueError:
        logger.warning("未知的 ctrl_motion: %s", ctrl_motion_value)
        ctrl_motion = MotionFsm.UNKNOWN

    try:
        ctrl_motion_state = ControlState(state_value)
    except ValueError:
        logger.warning("未知的 ctrl_motion_state: %s", state_value)
        ctrl_motion_state = ControlState.UNKNOWN

    return MotionInfo(ctrl_motion, ctrl_motion_state)
