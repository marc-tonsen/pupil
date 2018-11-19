import cv2
import numpy as np

from marker_tracker_3d.utils import split_param


def closest_angle_diff(vec_1, vec_2):
    vec_1 = np.array(vec_1).reshape(-1, 3)
    vec_2 = np.array(vec_2).reshape(-1, 3)
    diff = angle_between(vec_1, vec_2)
    diff_min = np.min(diff)

    return diff_min


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'
    Source: https://stackoverflow.com/questions/2827393/
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u.T), -1.0, 1.0))


def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector, axis=1)[:, np.newaxis]


def get_transform_mat(param):
    if param is None:
        return None
    rvec, tvec = split_param(param)
    mat_extrinsic = np.eye(4)
    mat_extrinsic[:3, :3] = cv2.Rodrigues(rvec)[0]
    mat_extrinsic[:3, 3] = tvec
    return mat_extrinsic


def get_camera_pose_mat(camera_params):
    camera_pose_inv = get_transform_mat(camera_params.copy())
    camera_pose = np.linalg.inv(camera_pose_inv)
    return camera_pose


def svdt(A, B, order="row"):
    """Calculates the transformation between two coordinate systems using SVD.
    This function determines the rotation matrix (R) and the translation vector
    (L) for a rigid body after the following transformation [1]_, [2]_:
    B = R*A + L + err.
    Where A and B represents the rigid body in different instants and err is an
    aleatory noise (which should be zero for a perfect rigid body). A and B are
    matrices with the marker coordinates at different instants (at least three
    non-collinear markers are necessary to determine the 3D transformation).
    The matrix A can be thought to represent a local coordinate system (but A
    it's not a basis) and matrix B the global coordinate system. The operation
    Pg = R*Pl + L calculates the coordinates of the point Pl (expressed in the
    local coordinate system) in the global coordinate system (Pg).
    A typical use of the svdt function is to calculate the transformation
    between A and B (B = R*A + L), where A is the matrix with the markers data
    in one instant (the calibration or static trial) and B is the matrix with
    the markers data for one or more instants (the dynamic trial).

    # __author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
    # __version__ = 'svdt.py v.1 2013/12/23'
    """

    A, B = np.asarray(A), np.asarray(B)
    if order == "row" or B.ndim == 1:
        if B.ndim == 1:
            A = A.reshape(A.size / 3, 3)
            B = B.reshape(B.size / 3, 3)
        R, L, RMSE = _svd(A, B)
    else:
        A = A.reshape(A.size / 3, 3)
        ni = B.shape[0]
        R = np.empty((ni, 3, 3))
        L = np.empty((ni, 3))
        RMSE = np.empty(ni)
        for i in range(ni):
            R[i, :, :], L[i, :], RMSE[i] = _svd(A, B[i, :].reshape(A.shape))

    return R, L, RMSE


def _svd(A, B):
    Am = np.mean(A, axis=0)  # centroid of m1
    Bm = np.mean(B, axis=0)  # centroid of m2
    M = np.dot((B - Bm).T, (A - Am))  # considering only rotation
    # singular value decomposition
    U, S, Vt = np.linalg.svd(M)
    # rotation matrix
    R = np.dot(U, np.dot(np.diag([1, 1, np.linalg.det(np.dot(U, Vt))]), Vt))
    # translation vector
    L = B.mean(0) - np.dot(R, A.mean(0))
    # RMSE
    err = 0
    for i in range(A.shape[0]):
        Bp = np.dot(R, A[i, :]) + L
        err += np.sum((Bp - B[i, :]) ** 2)
    RMSE = np.sqrt(err / A.shape[0] / 3)

    return R, L, RMSE
