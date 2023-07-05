# Add by Zhenhua Song
import numpy as np
from typing import Optional, Dict
from .MotionData import MotionData


def calc_children(data: MotionData):
    children = [[] for _ in range(data._num_joints)]
    for i, p in enumerate(data._skeleton_joint_parents[1:]):
        children[p].append(i + 1)
    return children


def calc_name_idx(data: MotionData) -> Dict[str, int]:
    return dict(zip(data.joint_names, range(len(data.joint_names))))


def adjust_root_height(data: MotionData, dh: Optional[float] = None):
    if dh is None:
        min_y_pos = np.min(data._joint_position[:, :, 1], axis=1)
        min_y_pos[min_y_pos > 0] = 0
        dy = np.min(min_y_pos)
    else:
        dy = dh
    data._joint_position[:, :, 1] -= dy
    data._joint_translation[:, 0, 1] -= dy
