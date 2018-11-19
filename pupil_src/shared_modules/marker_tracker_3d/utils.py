import os

import cv2
import numpy as np

import logging

logger = logging.getLogger(__name__)


def split_param(param):
    assert param.size == 6
    return param.ravel()[0:3], param.ravel()[3:6]


def merge_param(rvec, tvec):
    assert rvec.size == 3 and tvec.size == 3
    return np.concatenate((rvec.ravel(), tvec.ravel()))


def to_camera_coordinate(pts_3d_world, rvec, tvec):
    pts_3d_cam = [
        cv2.Rodrigues(rvec)[0] @ p + tvec.ravel() for p in pts_3d_world.reshape(-1, 3)
    ]
    pts_3d_cam = np.array(pts_3d_cam)

    return pts_3d_cam


def check_camera_param(pts_3d_world, rvec, tvec):
    if (np.abs(rvec) > np.pi * 2).any():
        return False

    pts_3d_camera = to_camera_coordinate(pts_3d_world, rvec, tvec)
    if (pts_3d_camera.reshape(-1, 3)[:, 2] < 1).any():
        return False

    return True


def save_params_dicts(save_path, dicts):
    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(os.path.join(save_path))
    for k, v in dicts.items():
        if isinstance(v, dict):
            save_dict_to_pkl(v, os.path.join(save_path, k))
        elif isinstance(v, np.ndarray) or isinstance(v, list):
            np.save(os.path.join(save_path, k), v)


def save_dict_to_pkl(d, dict_name):
    import pickle

    f = open(dict_name, "wb")
    pickle.dump(d, f)
    f.close()
