import itertools as it
import logging
import os
import collections

import networkx as nx
import numpy as np

from marker_tracker_3d.camera_model import CameraModel
from marker_tracker_3d.marker_model_3d import CameraLocalizer
from marker_tracker_3d.math import closest_angle_diff
from marker_tracker_3d.utils import split_param

logger = logging.getLogger(__name__)


class GraphForOptimization(CameraModel):
    def __init__(
        self,
        first_node_id=None,
        min_number_of_markers_per_frame_for_opt=3,
        min_number_of_frames_per_marker=2,
        min_camera_angle_diff=0.1,
        optimization_interval=1,
    ):
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
        self.marker_extrinsics_opt = collections.OrderedDict()

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
        assert isinstance(current_frame_new, dict), TypeError(
            "current_frame_new should be a dict"
        )
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

        self.marker_extrinsics_opt = {self.first_node: self.marker_extrinsics_origin}
        self.marker_keys = [self.first_node]
        # initialize self.marker_model_3d for _predict_camera_pose
        self.localization = CameraLocalizer(self.marker_extrinsics_opt)
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
            if n_id in self.visibility_graph_of_all_markers.nodes and len(
                self.visibility_graph_of_all_markers.nodes[n_id]
            ):
                diff = closest_angle_diff(
                    rvec,
                    list(self.visibility_graph_of_all_markers.nodes[n_id].values()),
                )
                if diff > self.min_angle_diff:
                    candidate_markers.append(n_id)
            else:
                candidate_markers.append(n_id)

        return candidate_markers

    def _decide_keyframe(self, candidate_markers, camera_params_loc):
        """
        decide if current_frame can be a keyframe
        add "previous_camera_extrinsics" as a key in the self.keyframes[self.frame_id] dicts
         """

        if len(candidate_markers) < self.min_number_of_markers_per_frame:
            return False

        self.keyframes[self.frame_id] = {
            k: v for k, v in self.current_frame.items() if k in candidate_markers
        }
        self.keyframes[self.frame_id]["previous_camera_extrinsics"] = camera_params_loc
        logger.debug(
            "--> keyframe {0}; markers {1}".format(self.frame_id, candidate_markers)
        )

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
            self.visibility_graph_of_ready_markers = (
                self.visibility_graph_of_all_markers.copy()
            )
            while True:
                # remove the nodes which are not viewed more than self.min_number_of_frames_per_marker times
                nodes_not_candidate = [
                    n
                    for n in self.visibility_graph_of_ready_markers.nodes
                    if len(self.visibility_graph_of_ready_markers.nodes[n])
                    < self.min_number_of_frames_per_marker
                ]
                self._remove_nodes(nodes_not_candidate)

                if (
                    len(self.visibility_graph_of_ready_markers) == 0
                    or self.first_node not in self.visibility_graph_of_ready_markers
                ):
                    return

                # remove the nodes which are not connected to the first node
                nodes_not_connected = list(
                    set(self.visibility_graph_of_ready_markers.nodes)
                    - set(
                        nx.node_connected_component(
                            self.visibility_graph_of_ready_markers, self.first_node
                        )
                    )
                )
                self._remove_nodes(nodes_not_connected)

                if len(self.visibility_graph_of_ready_markers) == 0:
                    return

                if len(nodes_not_candidate) == 0 and len(nodes_not_connected) == 0:
                    return

    def _remove_nodes(self, nodes):
        """ remove nodes in the graph """

        # remove the attribute of the node if the corresponding edges should be removed
        removed_edges = set(
            f
            for n1, n2, f in self.visibility_graph_of_ready_markers.edges(keys=True)
            if n1 in nodes or n2 in nodes
        )

        # remove the nodes
        self.visibility_graph_of_ready_markers.remove_nodes_from(nodes)

        for n_id in self.visibility_graph_of_ready_markers.nodes:
            for f_id in removed_edges:
                if f_id in self.visibility_graph_of_ready_markers.nodes[n_id]:
                    del self.visibility_graph_of_ready_markers.nodes[n_id][f_id]

    def _update_camera_and_marker_keys(self):
        """ add new ids to self.marker_keys """

        if self.first_node is not None:
            self.camera_keys = list(
                sorted(
                    set(
                        f_id
                        for _, _, f_id in self.visibility_graph_of_ready_markers.edges(
                            keys=True
                        )
                    )
                )
            )
            logger.debug("self.camera_keys updated {}".format(self.camera_keys))

            self.marker_keys = [self.first_node] + [
                n
                for n in self.visibility_graph_of_ready_markers.nodes
                if n != self.first_node
            ]
            logger.debug("self.marker_keys updated {}".format(self.marker_keys))

    def _prepare_data_for_optimization(self):
        """ prepare data for optimization """

        camera_indices, marker_indices, markers_points_2d_detected = (
            list(),
            list(),
            list(),
        )
        for f_id in self.camera_keys:
            for n_id in self.keyframes[f_id].keys() & set(self.marker_keys):
                camera_indices.append(self.camera_keys.index(f_id))
                marker_indices.append(self.marker_keys.index(n_id))
                markers_points_2d_detected.append(self.keyframes[f_id][n_id]["verts"])

        if len(markers_points_2d_detected):
            camera_indices = np.array(camera_indices)
            marker_indices = np.array(marker_indices)
            markers_points_2d_detected = np.array(markers_points_2d_detected)[
                :, :, 0, :
            ]
        else:
            return

        camera_params_prv = {}
        for i, k in enumerate(self.camera_keys):
            if k in self.camera_params_opt:
                camera_params_prv[i] = self.camera_params_opt[k]
            elif "previous_camera_extrinsics" in self.keyframes[k].keys():
                camera_params_prv[i] = self.keyframes[k][
                    "previous_camera_extrinsics"
                ].ravel()

        marker_extrinsics_prv = {}
        for i, k in enumerate(self.marker_keys):
            if k in self.marker_extrinsics_opt:
                marker_extrinsics_prv[i] = self.marker_extrinsics_opt[k]

        data_for_optimization = (
            camera_indices,
            marker_indices,
            markers_points_2d_detected,
            camera_params_prv,
            marker_extrinsics_prv,
        )

        return data_for_optimization

    def optimization_post_process(self, lock, result_opt_run):
        """ process the results of optimization """

        with lock:
            if isinstance(result_opt_run, dict) and len(result_opt_run) == 4:
                camera_params_opt = result_opt_run["camera_params_opt"]
                marker_extrinsics_opt = result_opt_run["marker_extrinsics_opt"]
                camera_index_failed = result_opt_run["camera_index_failed"]
                marker_index_failed = result_opt_run["marker_index_failed"]

                self._update_params_opt(
                    camera_params_opt,
                    marker_extrinsics_opt,
                    camera_index_failed,
                    marker_index_failed,
                )

                # remove those frame_id, which make optimization fail from self.keyframes
                self._discard_keyframes(camera_index_failed)

                self.camera_keys_prv = self.camera_keys.copy()

                return self.marker_extrinsics_opt

    def _update_params_opt(
        self, camera_params, marker_extrinsics, camera_index_failed, marker_index_failed
    ):
        for i, p in enumerate(camera_params):
            if i not in camera_index_failed:
                self.camera_params_opt[self.camera_keys[i]] = p
        for i, p in enumerate(marker_extrinsics):
            if i not in marker_index_failed:
                self.marker_extrinsics_opt[self.marker_keys[i]] = p
        logger.debug("update {}".format(self.marker_extrinsics_opt.keys()))

        for k in self.marker_keys:
            if k not in self.marker_keys_optimized and k in self.marker_extrinsics_opt:
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
        redundant_edges = [
            (n_id, neighbor, f_id)
            for n_id, neighbor, f_id in self.visibility_graph_of_all_markers.edges(
                keys=True
            )
            if f_id in failed_keyframes
        ]
        self.visibility_graph_of_all_markers.remove_edges_from(redundant_edges)

        # remove the attribute "previous_camera_extrinsics" of the node
        for f_id in failed_keyframes:
            for n_id in set(n for n, _, f in redundant_edges if f == f_id) | set(
                n for _, n, f in redundant_edges if f == f_id
            ):
                del self.visibility_graph_of_all_markers.nodes[n_id][f_id]

        fail_marker_keys = set(self.marker_keys) - set(
            self.marker_extrinsics_opt.keys()
        )
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

            nx.draw_networkx_nodes(
                graph_vis, pos, nodelist=all_nodes, node_color="g", node_size=100
            )
            if self.first_node in self.visibility_graph_of_ready_markers:
                connected_component = nx.node_connected_component(
                    self.visibility_graph_of_ready_markers, self.first_node
                )
                nx.draw_networkx_nodes(
                    graph_vis,
                    pos,
                    nodelist=connected_component,
                    node_color="r",
                    node_size=100,
                )
            nx.draw_networkx_edges(graph_vis, pos, width=1, alpha=0.1)
            nx.draw_networkx_labels(graph_vis, pos, font_size=7)

            labels = dict(
                (n, self.marker_keys.index(n) if n in self.marker_keys else None)
                for n in graph_vis.nodes()
            )
            nx.draw_networkx_labels(
                graph_vis, pos=pos_label, labels=labels, font_size=6, font_color="b"
            )

            plt.axis("off")
            save_name = os.path.join(
                save_path,
                "weighted_graph-{0:03d}-{1}-{2}-{3}.png".format(
                    self.frame_id,
                    len(self.visibility_graph_of_all_markers),
                    len(self.visibility_graph_of_ready_markers),
                    len(self.marker_keys_optimized),
                ),
            )
            plt.savefig(save_name)
            plt.clf()
