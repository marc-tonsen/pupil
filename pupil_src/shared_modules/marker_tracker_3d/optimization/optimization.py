import logging
import time

import cv2
import numpy as np
import scipy

from marker_tracker_3d.camera_model import CameraModel
from marker_tracker_3d.math import get_transform_mat
from marker_tracker_3d.utils import check_camera_param, merge_param

logger = logging.getLogger(__name__)


class Optimization(CameraModel):
    def __init__(
        self,
        camera_indices,
        marker_indices,
        markers_points_2d_detected,
        camera_params_prv,
        marker_extrinsics_prv,
    ):
        """
        :param camera_indices: array_like with shape (n, ), camera indices
        :param marker_indices: array_like with shape (n, ), marker indices
        :param markers_points_2d_detected: np.ndarray with shape (n x 4 x 2), markers points from image
        :param camera_params_prv: dict, previous camera params
        :param marker_extrinsics_prv: dict, previous marker params
        """

        super().__init__()
        self.camera_indices = camera_indices
        self.marker_indices = marker_indices
        self.markers_points_2d_detected = markers_points_2d_detected

        assert isinstance(camera_params_prv, dict)
        assert isinstance(marker_extrinsics_prv, dict)
        self.camera_params_prv = camera_params_prv
        self.marker_extrinsics_prv = marker_extrinsics_prv

        self.n_cameras = len(set(self.camera_indices))
        self.n_markers = len(
            set(self.marker_extrinsics_prv.keys()) | set(self.marker_indices)
        )

        self.tol = 1e-3
        self.diff_step = 1e-3
        self.result_opt_run = None

    def prepare_data_for_reconstruct_camera(self, marker_extrinsics_init, camera_idx):
        """ prepare data for reconstruction using in cv2.solvePnP() """

        marker_index_available = (
            set(self.marker_indices[self.camera_indices == camera_idx])
            & marker_extrinsics_init.keys()
        )
        if len(marker_index_available) == 0:
            return [], []
        marker_key = min(marker_index_available)

        marker_points_3d_for_rec = self.params_to_points_3d(
            marker_extrinsics_init[marker_key]
        )
        marker_points_2d_for_rec = self.markers_points_2d_detected[
            np.bitwise_and(
                self.camera_indices == camera_idx, self.marker_indices == marker_key
            )
        ]

        if len(marker_points_3d_for_rec) and len(marker_points_2d_for_rec):
            marker_points_3d_for_rec.shape = 1, -1, 3
            marker_points_2d_for_rec.shape = 1, -1, 2

        return marker_points_3d_for_rec, marker_points_2d_for_rec

    def _reconstruction(self):
        """ reconstruct camera params and markers params iteratively
        the results are used as the initial guess for bundle adjustment
        """

        camera_params_init = self.camera_params_prv.copy()
        marker_extrinsics_init = self.marker_extrinsics_prv.copy()
        camera_index_not_computed = set(self.camera_indices) - set(
            camera_params_init.keys()
        )
        marker_index_not_computed = set(self.marker_indices) - set(
            marker_extrinsics_init.keys()
        )

        for ii in range(3):
            # reconstruct cameras
            for camera_idx in camera_index_not_computed:
                marker_points_3d_for_rec, marker_points_2d_for_rec = self.prepare_data_for_reconstruct_camera(
                    marker_extrinsics_init, camera_idx
                )

                retval, rvec, tvec = self.run_solvePnP(
                    marker_points_3d_for_rec, marker_points_2d_for_rec
                )

                if retval:
                    if check_camera_param(marker_points_3d_for_rec, rvec, tvec):
                        camera_params_init[camera_idx] = merge_param(rvec, tvec)

            # reconstruct markers
            for marker_idx in marker_index_not_computed:
                camera_index_available = list(
                    set(camera_params_init.keys())
                    & set(self.camera_indices[self.marker_indices == marker_idx])
                )
                if len(camera_index_available) < 2:
                    continue
                camera_idx0, camera_idx1 = np.random.choice(
                    camera_index_available, 2, replace=False
                )

                # triangulate points
                marker_extrinsics_init[marker_idx] = self.run_triangulation(
                    camera_params_init, camera_idx0, camera_idx1, marker_idx
                )

            camera_index_not_computed = set(self.camera_indices) - set(
                camera_params_init.keys()
            )
            marker_index_not_computed = set(self.marker_indices) - set(
                marker_extrinsics_init.keys()
            )
            if (
                len(camera_index_not_computed) == 0
                and len(marker_index_not_computed) == 0
            ):
                break

        if len(camera_index_not_computed) > 0 or len(marker_index_not_computed) > 0:
            return [], []

        return camera_params_init, marker_extrinsics_init

    def run_triangulation(self, camera_params, camera_idx0, camera_idx1, marker_idx):
        proj_mat1 = get_transform_mat(camera_params[camera_idx0])[:3, :4]
        proj_mat2 = get_transform_mat(camera_params[camera_idx1])[:3, :4]

        points1 = self.markers_points_2d_detected[
            np.bitwise_and(
                self.camera_indices == camera_idx0, self.marker_indices == marker_idx
            )
        ]
        undistort_points1 = cv2.undistortPoints(
            points1, self.cameraMatrix, self.distCoeffs
        )

        points2 = self.markers_points_2d_detected[
            np.bitwise_and(
                self.camera_indices == camera_idx1, self.marker_indices == marker_idx
            )
        ]
        undistort_points2 = cv2.undistortPoints(
            points2, self.cameraMatrix, self.distCoeffs
        )

        points4D = cv2.triangulatePoints(
            proj_mat1, proj_mat2, undistort_points1, undistort_points2
        )
        marker_points_3d = cv2.convertPointsFromHomogeneous(points4D.T).reshape(4, 3)
        marker_extrinsics = self.point_3d_to_param(marker_points_3d)

        return marker_extrinsics

    def _find_sparsity(self):
        """
        Defines the sparsity structure of the Jacobian matrix for finite difference estimation.
        If the Jacobian has only few non-zero elements in each row, providing the sparsity structure will greatly speed
        up the computations. A zero entry means that a corresponding element in the Jacobian is identically zero.
        """

        n_residuals = self.markers_points_2d_detected.size
        n_params = (
            self.n_camera_params * self.n_cameras
            + self.n_marker_extrinsics * self.n_markers
        )
        logger.debug(
            "n_cameras {0} n_markers {1} n_residuals {2} n_params {3}".format(
                self.n_cameras, self.n_markers, n_residuals, n_params
            )
        )

        sparsity_mat = scipy.sparse.lil_matrix((n_residuals, n_params), dtype=int)
        i = np.arange(self.camera_indices.size)

        for s in range(self.n_camera_params):
            for j in range(8):
                sparsity_mat[
                    8 * i + j, self.camera_indices * self.n_camera_params + s
                ] = 1

        for s in range(self.n_marker_extrinsics):
            for j in range(8):
                sparsity_mat[
                    8 * i + j,
                    self.n_cameras * self.n_camera_params
                    + self.marker_indices * self.n_marker_extrinsics
                    + s,
                ] = 1

        return sparsity_mat

    # Fix first marker
    def _cal_bounds(self, x, epsilon=1e-8):
        """ calculate the lower and upper bounds on independent variables """

        camera_params_size = self.n_cameras * self.n_camera_params
        lower_bound = np.full_like(x, -np.inf)
        lower_bound[
            camera_params_size : camera_params_size + self.n_marker_extrinsics
        ] = (self.marker_extrinsics_origin - epsilon)
        upper_bound = np.full_like(x, np.inf)
        upper_bound[
            camera_params_size : camera_params_size + self.n_marker_extrinsics
        ] = (self.marker_extrinsics_origin + epsilon)
        assert (
            (x > lower_bound)[
                camera_params_size : camera_params_size + self.n_marker_extrinsics
            ]
        ).all(), "lower_bound hit"
        assert (
            (x < upper_bound)[
                camera_params_size : camera_params_size + self.n_marker_extrinsics
            ]
        ).all(), "upper_bound hit"

        return lower_bound, upper_bound

    def _func(self, params):
        """
        Function which computes the vector of residuals, with the signature fun(x, *args, **kwargs),
        i.e., the minimization proceeds with respect to its first argument.
        The argument x passed to this function is an ndarray of shape (n,)
        """

        camera_params, marker_extrinsics = self._reshape_params(params)
        return self.cal_proj_error(
            camera_params,
            marker_extrinsics,
            self.camera_indices,
            self.marker_indices,
            self.markers_points_2d_detected,
        )

    def _reshape_params(self, params):
        """ reshape camera_params and marker_extrinsics into original shape"""

        camera_params_size = self.n_cameras * self.n_camera_params
        camera_params = params[:camera_params_size].reshape(
            self.n_cameras, self.n_camera_params
        )
        marker_extrinsics = params[camera_params_size:].reshape(
            self.n_markers, self.n_marker_extrinsics
        )
        return camera_params, marker_extrinsics

    def bundle_adjustment(
        self, camera_params_init, marker_extrinsics_init, verbose=False
    ):
        """ run bundle adjustment given the result of reconstruction """

        # initial guess
        camera_params_0 = np.array(
            [camera_params_init[i] for i in sorted(camera_params_init.keys())]
        )
        marker_extrinsics_0 = np.array(
            [marker_extrinsics_init[i] for i in sorted(marker_extrinsics_init.keys())]
        )

        x0 = np.hstack((camera_params_0.ravel(), marker_extrinsics_0.ravel()))

        bounds = self._cal_bounds(x0)
        A = self._find_sparsity()

        t0 = time.time()
        # do bundle adjustment by scipy.optimize
        res = scipy.optimize.least_squares(
            self._func,
            x0,
            jac_sparsity=A,
            x_scale="jac",
            method="trf",
            bounds=bounds,
            diff_step=self.diff_step,
            ftol=self.tol,
            xtol=self.tol,
            gtol=self.tol,
            verbose=verbose,
        )
        t1 = time.time()
        logger.debug("bundle_adjustment took {0:.4f} seconds".format(t1 - t0))

        camera_params_opt, marker_extrinsics_opt = self._reshape_params(res.x)
        return camera_params_opt, marker_extrinsics_opt

    def run(self, event):
        """ run reconstruction and then bundle adjustment """

        # Reconstruction
        camera_params_init, marker_extrinsics_init = self._reconstruction()
        if len(camera_params_init) == 0 or len(marker_extrinsics_init) == 0:
            self.result_opt_run = None
            logger.debug("reconstruction failed")
            event.set()
            return
        logger.debug("reconstruction done")

        # bundle adjustment
        camera_params_opt, marker_extrinsics_opt = self.bundle_adjustment(
            camera_params_init, marker_extrinsics_init
        )
        camera_index_failed, marker_index_failed = self._success_check(
            camera_params_opt,
            marker_extrinsics_opt,
            self.camera_indices,
            self.marker_indices,
            self.markers_points_2d_detected,
            12,
        )

        self.result_opt_run = {
            "camera_params_opt": camera_params_opt,
            "marker_extrinsics_opt": marker_extrinsics_opt,
            "camera_index_failed": camera_index_failed,
            "marker_index_failed": marker_index_failed,
        }
        logger.debug("bundle adjustment done")
        event.set()
        return

    def _success_check(
        self,
        camera_params,
        marker_extrinsics,
        camera_indices,
        marker_indices,
        markers_points_2d_detected,
        thres=5,
    ):
        """ check if the result of optimization is reasonable """

        camera_params = camera_params.reshape(-1, self.n_camera_params)
        marker_extrinsics = marker_extrinsics.reshape(-1, self.n_marker_extrinsics)
        markers_points_3d = self.params_to_points_3d(marker_extrinsics)
        markers_points_2d_projected = self.project_markers(
            camera_params[camera_indices], markers_points_3d[marker_indices]
        )

        # check if the projected points are within reasonable range
        max_projected_points = np.max(np.abs(markers_points_2d_projected), axis=(1, 2))
        camera_index_failed = set(camera_indices[max_projected_points > 1e4])
        marker_index_failed = set(marker_indices[max_projected_points > 1e4])

        # check if the reprojection errors are small
        reprojection_errors = np.linalg.norm(
            (markers_points_2d_detected - markers_points_2d_projected), axis=2
        ).sum(axis=1)
        camera_index_failed |= set(camera_indices[reprojection_errors > thres])
        marker_index_failed |= set(marker_indices[reprojection_errors > thres])

        return camera_index_failed, marker_index_failed
