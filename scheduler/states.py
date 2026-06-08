from enum import Enum, auto


class RobotState(Enum):
    """机器人任务状态枚举"""
    GO_TO_LOCATION = auto()  # 从A移动到指定位置
    PICK_AND_PUT = auto()    # 捡垃圾(可重试) + 放入后方垃圾框
    GO_DOCKING = auto()      # 返回充电桩
    FINISHED = auto()        # 任务成功完成
    FAILED = auto()          # 任务失败
