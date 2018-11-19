import os

import cv2
import numpy as np

import logging

logger = logging.getLogger(__name__)


def filter_markers(markers):
    markers = [m for m in markers if m["id_confidence"] > 0.9]

    markers_id_all = set([m["id"] for m in markers])
    for marker_id in markers_id_all:
        markers_with_same_id = [m for m in markers if m["id"] == marker_id]
        if len(markers_with_same_id) > 2:
            markers = [m for m in markers if m["id"] != marker_id]
            logger.warning(
                "WARNING! Multiple markers with same id {} found!".format(marker_id)
            )
        elif len(markers_with_same_id) == 2:
            markers = _remove_duplicate(marker_id, markers, markers_with_same_id)

    marker_dict = {m["id"]: {k: v for k, v in m.items() if k != "id"} for m in markers}

    return marker_dict


def _remove_duplicate(m_id, markers, markers_with_same_id):
    dist = np.linalg.norm(
        np.array(markers_with_same_id[0]["centroid"])
        - np.array(markers_with_same_id[1]["centroid"])
    )
    # If two markers are very close, pick the bigger one. It may due to double detection
    if dist < 3:
        marker_small = min(markers_with_same_id, key=lambda x: x["perimeter"])
        markers = [
            m
            for m in markers
            if not (m["id"] == m_id and m["centroid"] == marker_small["centroid"])
        ]
    else:
        markers = [m for m in markers if m["id"] != m_id]
        logger.warning("WARNING! Multiple markers with same id {} found!".format(m_id))
    return markers


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
