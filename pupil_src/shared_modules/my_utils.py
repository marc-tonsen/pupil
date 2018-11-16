import numpy as np
import cv2
import scipy
import time
import os
import threading
import networkx as nx
import itertools as it

# logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# import camera_models


class MyCamModel:
    def __init__(self, cameraMatrix=None, distCoeffs=None):

        if cameraMatrix is None:
            self.cameraMatrix = np.array(
                [[829.3510515270362, 0.0, 659.9293047259697],
                 [0.0, 799.5709408845464, 373.0776462356668],
                 [0.0, 0.0, 1.0]])
        else:
            self.cameraMatrix = cameraMatrix
        if distCoeffs is None:
            self.distCoeffs = np.array([[-0.43738542863224966, 0.190570781428104, -0.00125233833830639,
                                         0.0018723428760170056, -0.039219091259637684]])
        else:
            self.distCoeffs = distCoeffs

        self.camera_def = np.array([
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [0, 1, 0],  # 2
            [0, 0, 1],  # 3
        ], dtype=np.float)
        self.marker_df = np.array([
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
        ], dtype=np.float)
        self.marker_df_h = cv2.convertPointsToHomogeneous(self.marker_df).reshape(4, 4)
        self.n_camera_params = 6
        self.n_marker_params = 6
        self.marker_params_origin = self.point_3d_to_param(self.marker_df)

    def project_markers(self, camera_params, markers_points_3d):
        camera_params = camera_params.reshape(-1, self.n_camera_params).copy()
        markers_points_3d = markers_points_3d.reshape(-1, 4, 3).copy()
        markers_points_2d_projected = [cv2.projectPoints(points, cam[0:3], cam[3:6], self.cameraMatrix, self.distCoeffs)[0]
                               for cam, points in zip(camera_params, markers_points_3d)]
        markers_points_2d_projected = np.array(markers_points_2d_projected, dtype=np.float32)[:, :, 0, :]
        return markers_points_2d_projected

    def params_to_points_3d(self, params):
        params = np.asarray(params).reshape(-1, self.n_marker_params)
        marker_points_3d = list()
        for param in params:
            rvec, tvec = split_param(param)
            mat = np.eye(4, dtype=np.float32)
            mat[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
            mat[0:3, 3] = tvec
            marker_transformed_h = mat @ self.marker_df_h.T
            marker_transformed = cv2.convertPointsFromHomogeneous(marker_transformed_h.T).reshape(4, 3)
            marker_points_3d.append(marker_transformed)

        marker_points_3d = np.array(marker_points_3d)
        return marker_points_3d

    def point_3d_to_param(self, marker_points_3d):
        R, L, RMSE = svdt(A=self.marker_df, B=marker_points_3d)
        rvec = cv2.Rodrigues(R)[0]
        tvec = L
        marker_params = merge_param(rvec, tvec)
        return marker_params

    def cal_cost(self, camera_params, marker_params, camera_indices, marker_indices, markers_points_2d_detected):
        residuals = self.cal_proj_error(camera_params, marker_params, camera_indices, marker_indices, markers_points_2d_detected)
        cost = 0.5 * np.sum(residuals ** 2)
        return cost

    def cal_proj_error(self, camera_params, marker_params, camera_indices, marker_indices, markers_points_2d_detected):
        markers_points_3d = self.params_to_points_3d(marker_params.reshape(-1, 6))
        markers_points_2d_projected = self.project_markers(camera_params[camera_indices], markers_points_3d[marker_indices])
        diff = (markers_points_2d_projected - markers_points_2d_detected)
        return diff.ravel()

    def run_solvePnP(self, marker_points_3d, marker_points_2d, camera_params_prv=None):
        if len(marker_points_3d) == 0 or len(marker_points_2d) == 0:
            return False, None, None

        if marker_points_3d.shape[1] != marker_points_2d.shape[1]:
            return False, None, None

        if camera_params_prv is not None:
            rvec, tvec = split_param(camera_params_prv)
            retval, rvec, tvec = cv2.solvePnP(marker_points_3d, marker_points_2d, self.cameraMatrix, self.distCoeffs,
                                              useExtrinsicGuess=True, rvec=rvec.copy(), tvec=tvec.copy())
        else:
            retval, rvec, tvec = cv2.solvePnP(marker_points_3d, marker_points_2d, self.cameraMatrix, self.distCoeffs)

        return retval, rvec, tvec


class Optimization(MyCamModel):
    def __init__(self, camera_indices, marker_indices, markers_points_2d_detected, camera_params_prv, marker_params_prv):
        """
        :param camera_indices: array_like with shape (n, ), camera indices
        :param marker_indices: array_like with shape (n, ), marker indices
        :param markers_points_2d_detected: np.ndarray with shape (n x 4 x 2), markers points from image
        :param camera_params_prv: dict, previous camera params
        :param marker_params_prv: dict, previous marker params
        """

        super().__init__()
        self.camera_indices = camera_indices
        self.marker_indices = marker_indices
        self.markers_points_2d_detected = markers_points_2d_detected

        assert isinstance(camera_params_prv, dict)
        assert isinstance(marker_params_prv, dict)
        self.camera_params_prv = camera_params_prv
        self.marker_params_prv = marker_params_prv

        self.n_cameras = len(set(self.camera_indices))
        self.n_markers = len(set(self.marker_params_prv.keys()) | set(self.marker_indices))

        self.tol = 1e-3
        self.diff_step = 1e-3
        self.result_opt_run = None

    def prepare_data_for_reconstruct_camera(self, marker_params_init, camera_idx):
        """ prepare data for reconstruction using in cv2.solvePnP() """

        marker_index_available = set(self.marker_indices[self.camera_indices == camera_idx]) & marker_params_init.keys()
        if len(marker_index_available) == 0:
            return [], []
        marker_key = min(marker_index_available)

        marker_points_3d_for_rec = self.params_to_points_3d(marker_params_init[marker_key])
        marker_points_2d_for_rec = self.markers_points_2d_detected[np.bitwise_and(self.camera_indices == camera_idx,
                                                                                          self.marker_indices == marker_key)]

        if len(marker_points_3d_for_rec) and len(marker_points_2d_for_rec):
            marker_points_3d_for_rec.shape = 1, -1, 3
            marker_points_2d_for_rec.shape = 1, -1, 2

        return marker_points_3d_for_rec, marker_points_2d_for_rec

    def _reconstruction(self):
        """ reconstruct camera params and markers params iteratively
        the results are used as the initial guess for bundle adjustment
        """

        camera_params_init = self.camera_params_prv.copy()
        marker_params_init = self.marker_params_prv.copy()
        camera_index_not_computed = set(self.camera_indices) - set(camera_params_init.keys())
        marker_index_not_computed = set(self.marker_indices) - set(marker_params_init.keys())

        for ii in range(3):
            # reconstruct cameras
            for camera_idx in camera_index_not_computed:
                marker_points_3d_for_rec, marker_points_2d_for_rec = self.prepare_data_for_reconstruct_camera(marker_params_init, camera_idx)

                retval, rvec, tvec = self.run_solvePnP(marker_points_3d_for_rec, marker_points_2d_for_rec)

                if retval:
                    if check_camera_param(marker_points_3d_for_rec, rvec, tvec):
                        camera_params_init[camera_idx] = merge_param(rvec, tvec)

            # reconstruct markers
            for marker_idx in marker_index_not_computed:
                camera_index_available = list(set(camera_params_init.keys()) & set(self.camera_indices[self.marker_indices == marker_idx]))
                if len(camera_index_available) < 2:
                    continue
                camera_idx0, camera_idx1 = np.random.choice(camera_index_available, 2, replace=False)

                # triangulate points
                marker_params_init[marker_idx] = self.run_triangulation(camera_params_init, camera_idx0, camera_idx1, marker_idx)

            camera_index_not_computed = set(self.camera_indices) - set(camera_params_init.keys())
            marker_index_not_computed = set(self.marker_indices) - set(marker_params_init.keys())
            if len(camera_index_not_computed) == 0 and len(marker_index_not_computed) == 0:
                break

        if len(camera_index_not_computed) > 0 or len(marker_index_not_computed) > 0:
            return [], []
            
        return camera_params_init, marker_params_init

    def run_triangulation(self, camera_params, camera_idx0, camera_idx1, marker_idx):
        proj_mat1 = get_transform_mat(camera_params[camera_idx0])[:3, :4]
        proj_mat2 = get_transform_mat(camera_params[camera_idx1])[:3, :4]

        points1 = self.markers_points_2d_detected[np.bitwise_and(self.camera_indices == camera_idx0, self.marker_indices == marker_idx)]
        undistort_points1 = cv2.undistortPoints(points1, self.cameraMatrix, self.distCoeffs)

        points2 = self.markers_points_2d_detected[np.bitwise_and(self.camera_indices == camera_idx1, self.marker_indices == marker_idx)]
        undistort_points2 = cv2.undistortPoints(points2, self.cameraMatrix, self.distCoeffs)

        points4D = cv2.triangulatePoints(proj_mat1, proj_mat2, undistort_points1, undistort_points2)
        marker_points_3d = cv2.convertPointsFromHomogeneous(points4D.T).reshape(4, 3)
        marker_params = self.point_3d_to_param(marker_points_3d)
        
        return marker_params
    
    def _find_sparsity(self):
        """
        Defines the sparsity structure of the Jacobian matrix for finite difference estimation.
        If the Jacobian has only few non-zero elements in each row, providing the sparsity structure will greatly speed
        up the computations. A zero entry means that a corresponding element in the Jacobian is identically zero.
        """

        n_residuals = self.markers_points_2d_detected.size
        n_params = self.n_camera_params * self.n_cameras + self.n_marker_params * self.n_markers
        logger.debug("n_cameras {0} n_markers {1} n_residuals {2} n_params {3}".format(self.n_cameras, self.n_markers, n_residuals, n_params))

        sparsity_mat = scipy.sparse.lil_matrix((n_residuals, n_params), dtype=int)
        i = np.arange(self.camera_indices.size)

        for s in range(self.n_camera_params):
            for j in range(8):
                sparsity_mat[8 * i + j, self.camera_indices * self.n_camera_params + s] = 1

        for s in range(self.n_marker_params):
            for j in range(8):
                sparsity_mat[8 * i + j, self.n_cameras * self.n_camera_params + self.marker_indices * self.n_marker_params + s] = 1

        return sparsity_mat

    # Fix first marker
    def _cal_bounds(self, x, epsilon=1e-8):
        """ calculate the lower and upper bounds on independent variables """

        camera_params_size = self.n_cameras * self.n_camera_params
        lower_bound = np.full_like(x, -np.inf)
        lower_bound[camera_params_size: camera_params_size + self.n_marker_params] = self.marker_params_origin - epsilon
        upper_bound = np.full_like(x, np.inf)
        upper_bound[camera_params_size: camera_params_size + self.n_marker_params] = self.marker_params_origin + epsilon
        assert ((x > lower_bound)[camera_params_size: camera_params_size + self.n_marker_params]).all(), "lower_bound hit"
        assert ((x < upper_bound)[camera_params_size: camera_params_size + self.n_marker_params]).all(), "upper_bound hit"

        return lower_bound, upper_bound

    def _func(self, params):
        """
        Function which computes the vector of residuals, with the signature fun(x, *args, **kwargs),
        i.e., the minimization proceeds with respect to its first argument.
        The argument x passed to this function is an ndarray of shape (n,)
        """

        camera_params, marker_params = self._reshape_params(params)
        return self.cal_proj_error(camera_params, marker_params, self.camera_indices, self.marker_indices, self.markers_points_2d_detected)

    def _reshape_params(self, params):
        """ reshape camera_params and marker_params into original shape"""

        camera_params_size = self.n_cameras * self.n_camera_params
        camera_params = params[:camera_params_size].reshape(self.n_cameras, self.n_camera_params)
        marker_params = params[camera_params_size:].reshape(self.n_markers, self.n_marker_params)
        return camera_params, marker_params

    def bundle_adjustment(self, camera_params_init, marker_params_init, verbose=False):
        """ run bundle adjustment given the result of reconstruction """

        # initial guess
        camera_params_0 = np.array([camera_params_init[i] for i in sorted(camera_params_init.keys())])
        marker_params_0 = np.array([marker_params_init[i] for i in sorted(marker_params_init.keys())])

        x0 = np.hstack((camera_params_0.ravel(), marker_params_0.ravel()))

        bounds = self._cal_bounds(x0)
        A = self._find_sparsity()

        t0 = time.time()
        # do bundle adjustment by scipy.optimize
        res = scipy.optimize.least_squares(self._func, x0, jac_sparsity=A, x_scale="jac", method="trf",
                                       bounds=bounds,
                                       diff_step=self.diff_step,
                                       ftol=self.tol, xtol=self.tol, gtol=self.tol,
                                       verbose=verbose)
        t1 = time.time()
        logger.debug("bundle_adjustment took {0:.4f} seconds".format(t1 - t0))

        camera_params_opt, marker_params_opt = self._reshape_params(res.x)
        return camera_params_opt, marker_params_opt

    def run(self, event):
        """ run reconstruction and then bundle adjustment """

        # Reconstruction
        camera_params_init, marker_params_init = self._reconstruction()
        if len(camera_params_init) == 0 or len(marker_params_init) == 0:
            self.result_opt_run = None
            logger.debug("reconstruction failed")
            event.set()
            return
        logger.debug("reconstruction done")

        # bundle adjustment
        camera_params_opt, marker_params_opt = self.bundle_adjustment(camera_params_init, marker_params_init)
        camera_index_failed, marker_index_failed = self._success_check(camera_params_opt, marker_params_opt, self.camera_indices, self.marker_indices, self.markers_points_2d_detected, 12)

        self.result_opt_run = {
            "camera_params_opt": camera_params_opt,
            "marker_params_opt": marker_params_opt,
            "camera_index_failed": camera_index_failed,
            "marker_index_failed": marker_index_failed,
        }
        logger.debug("bundle adjustment done")
        event.set()
        return

    def _success_check(self, camera_params, marker_params, camera_indices, marker_indices, markers_points_2d_detected, thres=5):
        """ check if the result of optimization is reasonable """

        camera_params = camera_params.reshape(-1, self.n_camera_params)
        marker_params = marker_params.reshape(-1, self.n_marker_params)
        markers_points_3d = self.params_to_points_3d(marker_params)
        markers_points_2d_projected = self.project_markers(camera_params[camera_indices], markers_points_3d[marker_indices])

        # check if the projected points are within reasonable range
        max_projected_points = np.max(np.abs(markers_points_2d_projected), axis=(1, 2))
        camera_index_failed = set(camera_indices[max_projected_points > 1e4])
        marker_index_failed = set(marker_indices[max_projected_points > 1e4])

        # check if the reprojection errors are small
        reprojection_errors = np.linalg.norm((markers_points_2d_detected - markers_points_2d_projected), axis=2).sum(axis=1)
        camera_index_failed |= set(camera_indices[reprojection_errors > thres])
        marker_index_failed |= set(marker_indices[reprojection_errors > thres])

        return camera_index_failed, marker_index_failed


class Localization(MyCamModel):
    def __init__(self, marker_params_opt):
        super().__init__()
        self.marker_params_opt = marker_params_opt

    @property
    def marker_params_opt(self):
        return self.__marker_params_opt

    @marker_params_opt.setter
    def marker_params_opt(self, marker_params_new):
        assert isinstance(marker_params_new, dict)
        self.__marker_params_opt = marker_params_new

    def prepare_data_for_localization(self, current_frame):
        marker_keys_available = current_frame.keys() & set(self.marker_params_opt.keys())

        marker_points_3d_for_loc = self.params_to_points_3d([self.marker_params_opt[i] for i in marker_keys_available])
        marker_points_2d_for_loc = np.array([current_frame[i]["verts"] for i in marker_keys_available])

        if len(marker_points_3d_for_loc) and len(marker_points_2d_for_loc):
            marker_points_3d_for_loc.shape = 1, -1, 3
            marker_points_2d_for_loc.shape = 1, -1, 2

        return marker_points_3d_for_loc, marker_points_2d_for_loc

    def current_camera(self, current_frame, camera_params_loc_prv=None):
        marker_points_3d_for_loc, marker_points_2d_for_loc = self.prepare_data_for_localization(current_frame)

        retval, rvec, tvec = self.run_solvePnP(marker_points_3d_for_loc, marker_points_2d_for_loc, camera_params_loc_prv)

        if retval:
            if check_camera_param(marker_points_3d_for_loc, rvec, tvec):
                camera_params_loc = merge_param(rvec, tvec)
                return camera_params_loc


class GraphForOptimization(MyCamModel):
    def __init__(self, first_node_id=None, min_number_of_markers_per_frame_for_opt=3, min_number_of_frames_per_marker=2,
                 min_camera_angle_diff=0.1, optimization_interval=1):
        super().__init__()
        assert min_number_of_markers_per_frame_for_opt >= 2
        assert min_number_of_frames_per_marker >= 2
        assert min_camera_angle_diff > 0
        assert optimization_interval >= 1

        self.min_number_of_markers_per_frame = min_number_of_markers_per_frame_for_opt
        self.min_number_of_frames_per_marker = min_number_of_frames_per_marker
        self.min_angle_diff = min_camera_angle_diff
        self.optimization_interval = optimization_interval

        self.current_frame = dict()
        self.frame_id = 0
        self.count_opt = 0

        self.marker_keys = list()
        self.marker_keys_optimized = list()
        self.camera_keys = list()
        self.camera_keys_prv = list()

        self.camera_params_opt = dict()
        self.marker_params_opt = dict()

        self.data_for_optimization = None
        self.localization = None

        self.keyframes = dict()
        self.first_node_id = first_node_id
        self.first_node = None
        self.visibility_graph_of_all_markers = nx.MultiGraph()
        self.visibility_graph_of_ready_markers = nx.MultiGraph()
        logger.debug("create MultiGraph")

    @property
    def current_frame(self):
        return self.__current_frame

    @current_frame.setter
    def current_frame(self, current_frame_new):
        assert isinstance(current_frame_new, dict), TypeError("current_frame_new should be a dict")
        if len(current_frame_new) >= self.min_number_of_markers_per_frame:
            self.__current_frame = current_frame_new
        else:
            self.__current_frame = dict()

    def update_visibility_graph_of_keyframes(self, lock, data):
        """ pick up keyframe and update visibility graph of keyframes """

        with lock:
            assert isinstance(data, tuple) and len(data) == 2
            current_frame, camera_params_loc = data
            self.current_frame = current_frame

            if len(self.current_frame) == 0:
                return

            if not self._find_first_node():
                return

            if camera_params_loc is None:
                camera_params_loc = self._predict_camera_pose()
                if camera_params_loc is None:
                    return

            candidate_markers = self._get_candidate_markers(camera_params_loc)
            if self._decide_keyframe(candidate_markers, camera_params_loc):
                self._add_to_graph(candidate_markers, camera_params_loc)
                self.count_opt += 1
            self.frame_id += 1

    def _find_first_node(self):
        if self.first_node is not None:
            return True

        if self.first_node_id is not None:
            if self.first_node_id in self.current_frame:
                self.first_node = self.first_node_id
            else:
                return False
        else:
            self.first_node = list(self.current_frame.keys())[0]

        self.marker_params_opt = {self.first_node: self.marker_params_origin}
        self.marker_keys = [self.first_node]
        # initialize self.localization for _predict_camera_pose
        self.localization = Localization(self.marker_params_opt)
        return True

    def _predict_camera_pose(self):
        """ predict current camera pose """

        if self.localization is not None:
            camera_params_tmp = self.localization.current_camera(self.current_frame)
            return camera_params_tmp

    def _get_candidate_markers(self, camera_params_loc):
        """
        get those markers in current_frame, to which the rotation vector of the current camera pose is diverse enough
        """

        rvec, _ = split_param(camera_params_loc)

        candidate_markers = list()
        for n_id in self.current_frame:
            if n_id in self.visibility_graph_of_all_markers.nodes and len(self.visibility_graph_of_all_markers.nodes[n_id]):
                diff = closest_angle_diff(rvec, list(self.visibility_graph_of_all_markers.nodes[n_id].values()))
                if diff > self.min_angle_diff:
                    candidate_markers.append(n_id)
            else:
                candidate_markers.append(n_id)

        return candidate_markers

    def _decide_keyframe(self, candidate_markers, camera_params_loc):
        """
        decide if current_frame can be a keyframe
        add "camera_params_loc" as a key in the self.keyframes[self.frame_id] dicts
         """

        if len(candidate_markers) < self.min_number_of_markers_per_frame:
            return False

        self.keyframes[self.frame_id] = {k: v for k, v in self.current_frame.items() if k in candidate_markers}
        self.keyframes[self.frame_id]["camera_params_loc"] = camera_params_loc
        logger.debug("--> keyframe {0}; markers {1}".format(self.frame_id, candidate_markers))

        return True

    def _add_to_graph(self, unique_marker_id, camera_params_loc):
        """
        graph"s node: marker id; attributes: the keyframe id
        graph"s edge: keyframe id, where two markers shown in the same frame
        """

        # add frame_id as edges in the graph
        for u, v in list(it.combinations(unique_marker_id, 2)):
            self.visibility_graph_of_all_markers.add_edge(u, v, key=self.frame_id)

        # add frame_id as an attribute of the node
        rvec, _ = split_param(camera_params_loc)
        for n_id in unique_marker_id:
            self.visibility_graph_of_all_markers.nodes[n_id][self.frame_id] = rvec

    def optimization_pre_process(self, lock):
        with lock:
            # Do optimization when there are some new keyframes selected
            if self.count_opt >= self.optimization_interval:
                self.count_opt = 0

                self._update_visibility_graph_of_ready_markers()
                self._update_camera_and_marker_keys()

                # prepare data for optimization
                data_for_optimization = self._prepare_data_for_optimization()
                return data_for_optimization

    def _update_visibility_graph_of_ready_markers(self):
        """
        find out ready markers for optimization
        """

        if self.first_node is not None:
            self.visibility_graph_of_ready_markers = self.visibility_graph_of_all_markers.copy()
            while True:
                # remove the nodes which are not viewed more than self.min_number_of_frames_per_marker times
                nodes_not_candidate = [n for n in self.visibility_graph_of_ready_markers.nodes if len(self.visibility_graph_of_ready_markers.nodes[n]) < self.min_number_of_frames_per_marker]
                self._remove_nodes(nodes_not_candidate)

                if len(self.visibility_graph_of_ready_markers) == 0 or self.first_node not in self.visibility_graph_of_ready_markers:
                    return

                # remove the nodes which are not connected to the first node
                nodes_not_connected = list(set(self.visibility_graph_of_ready_markers.nodes) - set(nx.node_connected_component(self.visibility_graph_of_ready_markers, self.first_node)))
                self._remove_nodes(nodes_not_connected)

                if len(self.visibility_graph_of_ready_markers) == 0:
                    return

                if len(nodes_not_candidate) == 0 and len(nodes_not_connected) == 0:
                    return

    def _remove_nodes(self, nodes):
        """ remove nodes in the graph """

        # remove the attribute of the node if the corresponding edges should be removed
        removed_edges = set(f for n1, n2, f in self.visibility_graph_of_ready_markers.edges(keys=True) if n1 in nodes or n2 in nodes)

        # remove the nodes
        self.visibility_graph_of_ready_markers.remove_nodes_from(nodes)

        for n_id in self.visibility_graph_of_ready_markers.nodes:
            for f_id in removed_edges:
                if f_id in self.visibility_graph_of_ready_markers.nodes[n_id]:
                    del self.visibility_graph_of_ready_markers.nodes[n_id][f_id]

    def _update_camera_and_marker_keys(self):
        """ add new ids to self.marker_keys """
        
        if self.first_node is not None:
            self.camera_keys = list(sorted(set(f_id for _, _, f_id in self.visibility_graph_of_ready_markers.edges(keys=True))))
            logger.debug("self.camera_keys updated {}".format(self.camera_keys))

            self.marker_keys = [self.first_node] + [n for n in self.visibility_graph_of_ready_markers.nodes if n != self.first_node]
            logger.debug("self.marker_keys updated {}".format(self.marker_keys))

    def _prepare_data_for_optimization(self):
        """ prepare data for optimization """

        camera_indices, marker_indices, markers_points_2d_detected = list(), list(), list()
        for f_id in self.camera_keys:
            for n_id in self.keyframes[f_id].keys() & set(self.marker_keys):
                camera_indices.append(self.camera_keys.index(f_id))
                marker_indices.append(self.marker_keys.index(n_id))
                markers_points_2d_detected.append(self.keyframes[f_id][n_id]["verts"])

        if len(markers_points_2d_detected):
            camera_indices = np.array(camera_indices)
            marker_indices = np.array(marker_indices)
            markers_points_2d_detected = np.array(markers_points_2d_detected)[:, :, 0, :]
        else:
            return

        camera_params_prv = {}
        for i, k in enumerate(self.camera_keys):
            if k in self.camera_params_opt:
                camera_params_prv[i] = self.camera_params_opt[k]
            elif "camera_params_loc" in self.keyframes[k].keys():
                camera_params_prv[i] = self.keyframes[k]["camera_params_loc"].ravel()

        marker_params_prv = {}
        for i, k in enumerate(self.marker_keys):
            if k in self.marker_params_opt:
                marker_params_prv[i] = self.marker_params_opt[k]

        data_for_optimization = (camera_indices, marker_indices, markers_points_2d_detected, camera_params_prv, marker_params_prv)

        return data_for_optimization

    def optimization_post_process(self, lock, result_opt_run):
        """ process the results of optimization """

        with lock:
            if isinstance(result_opt_run, dict) and len(result_opt_run) == 4:
                camera_params_opt = result_opt_run["camera_params_opt"]
                marker_params_opt = result_opt_run["marker_params_opt"]
                camera_index_failed = result_opt_run["camera_index_failed"]
                marker_index_failed = result_opt_run["marker_index_failed"]

                self._update_params_opt(camera_params_opt, marker_params_opt, camera_index_failed, marker_index_failed)

                # remove those frame_id, which make optimization fail from self.keyframes
                self._discard_keyframes(camera_index_failed)

                self.camera_keys_prv = self.camera_keys.copy()
                marker_params = np.array([self.marker_params_opt[k] for k in self.marker_keys_optimized])
                marker_points_3d = self.params_to_points_3d(marker_params)

                return self.marker_params_opt, marker_points_3d, self.first_node

    def _update_params_opt(self, camera_params, marker_params, camera_index_failed, marker_index_failed):
        for i, p in enumerate(camera_params):
            if i not in camera_index_failed:
                self.camera_params_opt[self.camera_keys[i]] = p
        for i, p in enumerate(marker_params):
            if i not in marker_index_failed:
                self.marker_params_opt[self.marker_keys[i]] = p
        logger.debug("update {}".format(self.marker_params_opt.keys()))

        for k in self.marker_keys:
            if k not in self.marker_keys_optimized and k in self.marker_params_opt:
                self.marker_keys_optimized.append(k)
        logger.debug("self.marker_keys_optimized {}".format(self.marker_keys_optimized))

    def _discard_keyframes(self, camera_index_failed):
        """ if the optimization failed, update keyframes, the graph """

        if len(camera_index_failed) == 0:
            return
        failed_keyframes = set(self.camera_keys[i] for i in camera_index_failed)
        logger.debug("remove from keyframes: {}".format(failed_keyframes))

        # remove the last keyframes
        for f_id in failed_keyframes:
            try:
                del self.keyframes[f_id]
            except KeyError:
                logger.debug("{} is not in keyframes".format(f_id))

        # remove edges (failed frame_id) from graph
        redundant_edges = [(n_id, neighbor, f_id) for n_id, neighbor, f_id in self.visibility_graph_of_all_markers.edges(keys=True) if f_id in failed_keyframes]
        self.visibility_graph_of_all_markers.remove_edges_from(redundant_edges)

        # remove the attribute "camera_params_loc" of the node
        for f_id in failed_keyframes:
            for n_id in set(n for n, _, f in redundant_edges if f == f_id) | set(n for _, n, f in redundant_edges if f == f_id):
                del self.visibility_graph_of_all_markers.nodes[n_id][f_id]

        fail_marker_keys = set(self.marker_keys) - set(self.marker_params_opt.keys())
        for k in fail_marker_keys:
            self.marker_keys.remove(k)
        logger.debug("remove from marker_keys: {}".format(fail_marker_keys))

    # For debug
    def vis_graph(self, save_path):
        import matplotlib.pyplot as plt

        if len(self.visibility_graph_of_all_markers) and self.first_node is not None:
            graph_vis = self.visibility_graph_of_all_markers.copy()
            all_nodes = list(graph_vis.nodes)

            pos = nx.spring_layout(graph_vis, seed=0)  # positions for all nodes
            pos_label = dict((n, pos[n] + 0.05) for n in pos)

            nx.draw_networkx_nodes(graph_vis, pos, nodelist=all_nodes, node_color="g", node_size=100)
            if self.first_node in self.visibility_graph_of_ready_markers:
                connected_component = nx.node_connected_component(self.visibility_graph_of_ready_markers, self.first_node)
                nx.draw_networkx_nodes(graph_vis, pos, nodelist=connected_component, node_color="r", node_size=100)
            nx.draw_networkx_edges(graph_vis, pos, width=1, alpha=0.1)
            nx.draw_networkx_labels(graph_vis, pos, font_size=7)

            labels = dict((n, self.marker_keys.index(n) if n in self.marker_keys else None) for n in graph_vis.nodes())
            nx.draw_networkx_labels(graph_vis, pos=pos_label, labels=labels, font_size=6, font_color="b")

            plt.axis("off")
            save_name = os.path.join(save_path, "weighted_graph-{0:03d}-{1}-{2}-{3}.png".format(
                self.frame_id, len(self.visibility_graph_of_all_markers), len(self.visibility_graph_of_ready_markers),
                len(self.marker_keys_optimized)))
            plt.savefig(save_name)
            plt.clf()


def get_current_frame(markers):
    markers = [m for m in markers if m["id_confidence"] > 0.9]
    markers_id_all = [m["id"] for m in markers]
    for m_id in set(markers_id_all):
        markers_tmp = [m for m in markers if m["id"] == m_id]
        if len(markers_tmp) > 2:
            markers = [m for m in markers if m["id"] != m_id]
            logger.warning("WARNING! Multiple markers with same id {} found!".format(m_id))
        elif len(markers_tmp) == 2:
            dist = np.linalg.norm(np.array(markers_tmp[0]["centroid"]) - np.array(markers_tmp[1]["centroid"]))
            # If two markers are very close, pick the bigger one. It may due to double detection
            if dist < 3:
                marker_small = min(markers_tmp, key=lambda x: x["perimeter"])
                markers = [m for m in markers if not (m["id"] == m_id and m["centroid"] == marker_small["centroid"])]
            else:
                markers = [m for m in markers if m["id"] != m_id]
                logger.warning("WARNING! Multiple markers with same id {} found!".format(m_id))

    current_frame = {m["id"]: {k: v for k, v in m.items() if k != "id"} for m in markers}

    return current_frame


def generator_optimization(recv_pipe):
    """ background process """

    first_node_id = None
    graph_for_optimization = GraphForOptimization(first_node_id=first_node_id)
    event_opt_done = threading.Event()
    event_opt_not_running = threading.Event()
    event_opt_not_running.set()
    lock = threading.RLock()

    while True:
        if recv_pipe.poll(0.001):
            msg, data_recv = recv_pipe.recv()
            if msg == "frame":
                graph_for_optimization.update_visibility_graph_of_keyframes(lock, data_recv)

            elif msg == "restart":
                graph_for_optimization = GraphForOptimization(first_node_id=first_node_id)
                event_opt_done = threading.Event()
                event_opt_not_running = threading.Event()
                event_opt_not_running.set()
                lock = threading.RLock()

            # for experiments
            elif msg == "save":
                dicts = {
                    "marker_params_opt": graph_for_optimization.marker_params_opt,
                    "camera_params_opt": graph_for_optimization.camera_params_opt,
                }
                save_path = data_recv
                save_params_dicts(save_path=save_path, dicts=dicts)
                graph_for_optimization.vis_graph(save_path)

        if event_opt_not_running.wait(0.0001):
            event_opt_not_running.clear()
            data_for_optimization = graph_for_optimization.optimization_pre_process(lock)
            if data_for_optimization is not None:
                opt = Optimization(*data_for_optimization)
                # move Optimization to another thread
                t1 = threading.Thread(name="opt_run", target=opt.run, args=(event_opt_done, ))
                t1.start()
            else:
                event_opt_not_running.set()

        if event_opt_done.wait(0.0001):
            event_opt_done.clear()
            result = graph_for_optimization.optimization_post_process(lock, opt.result_opt_run)
            event_opt_not_running.set()
            if isinstance(result, tuple) and len(result) == 3:
                yield "done", result


"""
utility functions
"""


def to_camera_coordinate(pts_3d_world, rvec, tvec):
    pts_3d_cam = [cv2.Rodrigues(rvec)[0] @ p + tvec.ravel() for p in pts_3d_world.reshape(-1, 3)]
    pts_3d_cam = np.array(pts_3d_cam)

    return pts_3d_cam


def check_camera_param(pts_3d_world, rvec, tvec):
    if (np.abs(rvec) > np.pi * 2).any():
        return False

    pts_3d_camera = to_camera_coordinate(pts_3d_world, rvec, tvec)
    if (pts_3d_camera.reshape(-1, 3)[:, 2] < 1).any():
        return False

    return True


def split_param(param):
    assert param.size == 6
    return param.ravel()[0:3], param.ravel()[3:6]


def merge_param(rvec, tvec):
    assert rvec.size == 3 and tvec.size == 3
    return np.concatenate((rvec.ravel(), tvec.ravel()))


"""
math functions
"""


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
    if camera_params is None:
        return None
    camera_pose_inv = get_transform_mat(camera_params.copy())
    camera_pose = np.linalg.inv(camera_pose_inv)
    return camera_pose


def svdt(A, B, order='row'):
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
    if order == 'row' or B.ndim == 1:
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


"""
For experiments
"""


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
