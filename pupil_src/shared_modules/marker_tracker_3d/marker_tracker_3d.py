"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections
import datetime
import logging
import multiprocessing as mp
import os

import numpy as np

import background_helper as bh
import marker_tracker_3d.generator_optimization
import marker_tracker_3d.math
import marker_tracker_3d.utils
from marker_tracker_3d.marker_detector import MarkerDetector
from marker_tracker_3d.marker_model_3d import CameraLocalizer
from marker_tracker_3d.user_interface import UserInterface
from marker_tracker_3d.camera_model import CameraModel
from plugin import Plugin

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)


class Marker_Tracker_3D(Plugin):
    """
    This plugin tracks the pose of the scene camera based on fiducial markers in the environment.
    """

    icon_chr = chr(0xEC07)
    icon_font = "pupil_icons"

    def __init__(self, g_pool, open_3d_window=True):
        super().__init__(g_pool)

        self.marker_detector = MarkerDetector()
        self.ui = UserInterface(self, open_3d_window)
        self.camera_model = CameraModel()

        # for tracking
        self.send_data_interval = 6
        self.min_number_of_markers_per_frame_for_loc = 2
        self.register_new_markers = True
        self.markers = {}
        self.marker_model_3d = CameraLocalizer()
        self.camera_trace = collections.deque(maxlen=100)
        self.camera_extrinsics = None
        self.previous_camera_extrinsics = None
        self.first_node = None
        self.frame_count = 0
        self.frame_count_last_send_data = 0

        # background process
        recv_pipe, self.send_pipe = mp.Pipe(False)
        generator_args = (recv_pipe,)
        self.bg_task = bh.IPC_Logging_Task_Proxy(
            name="generator_optimization",
            generator=marker_tracker_3d.generator_optimization.generator_optimization,
            args=generator_args,
        )

        # for experiments
        now = datetime.datetime.now()
        now_str = "%02d%02d%02d-%02d%02d" % (
            now.year,
            now.month,
            now.day,
            now.hour,
            now.minute,
        )
        self.save_path = os.path.join(
            "/cluster/users/Ching/experiments/marker_tracker_3d", now_str
        )
        self.robustness = list()
        self.camera_trace_all = list()
        self.all_frames = list()
        self.reprojection_errors = list()

    def init_ui(self):
        self.ui.init_ui()

    def deinit_ui(self):
        self.ui.deinit_ui()

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """

        self.close_window()
        if self.bg_task:
            self.bg_task.cancel()
            self.bg_task = None

    def restart(self):
        logger.warning("Restart!")
        self.reset_parameters()
        self.ui.update_menu()
        self.send_pipe.send(("restart", None))

    def save_data(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        dist = [
            np.linalg.norm(self.camera_trace_all[i + 1] - self.camera_trace_all[i])
            if self.camera_trace_all[i + 1] is not None
            and self.camera_trace_all[i] is not None
            else np.nan
            for i in range(len(self.camera_trace_all) - 1)
        ]

        dicts = {
            "dist": dist,
            "all_frames": self.all_frames,
            "reprojection_errors": self.reprojection_errors,
        }
        marker_tracker_3d.utils.save_params_dicts(save_path=self.save_path, dicts=dicts)

        self.send_pipe.send(("save", self.save_path))
        logger.info("save_data at {}".format(self.save_path))

    def reset_parameters(self):
        # for tracking
        self.register_new_markers = True
        self.markers = list()
        self.marker_model_3d = None
        self.camera_trace = collections.deque(maxlen=100)
        self.previous_camera_extrinsics = None
        self.first_node = None
        self.frame_count = 0
        self.frame_count_last_send_data = 0

        # for experiments
        self.robustness = list()
        self.camera_trace_all = list()
        self.all_frames = list()
        self.reprojection_errors = list()

    def get_init_dict(self):
        d = super().get_init_dict()
        d["open_3d_window"] = self.open_3d_window
        return d

    def recent_events(self, events):
        frame = events.get("frame")
        if not frame:
            self.early_exit()
            return

        self.fetch_marker_model_data_from_bg()
        self.markers = self.marker_detector.detect(frame)

        if len(self.markers) < self.min_number_of_markers_per_frame_for_loc:
            self.early_exit()
            return

        if self.marker_model_3d is not None:
            self.update_camera_extrinsics()

        self.send_marker_data_to_bg(camera_extrinsics=self.previous_camera_extrinsics)

        self.frame_count += 1

    def early_exit(self):
        if len(self.camera_trace):
            self.camera_trace.popleft()

    def fetch_marker_model_data_from_bg(self):
        for marker_extrinsics in self.bg_task.fetch():
            if marker_extrinsics:
                self.ui.update_menu()

            try:
                self.marker_model_3d.marker_extrinsics = marker_extrinsics
            except AttributeError:
                self.marker_model_3d = CameraLocalizer(marker_extrinsics)

            logger.info(
                "{} markers have been registered and updated".format(
                    len(marker_extrinsics)
                )
            )

    def send_marker_data_to_bg(self, camera_extrinsics):
        if (
            self.frame_count - self.frame_count_last_send_data
            >= self.send_data_interval
        ):
            self.frame_count_last_send_data = self.frame_count
            if self.register_new_markers:
                self.send_pipe.send(("frame", (self.markers, camera_extrinsics)))

    def update_camera_extrinsics(self):
        self.camera_extrinsics = self.marker_model_3d.current_camera(
            self.markers, self.previous_camera_extrinsics
        )
        if self.camera_extrinsics is None:
            # Do not set previous_camera_extrinsics to None to ensure a decent initial
            # guess for the next solve_pnp call
            self.camera_trace.append(None)
            self.camera_trace_all.append(None)
        else:
            self.previous_camera_extrinsics = self.camera_extrinsics

            camera_pose_matrix = marker_tracker_3d.math.get_camera_pose_mat(
                self.camera_extrinsics
            )
            self.camera_trace.append(camera_pose_matrix[0:3, 3])
            self.camera_trace_all.append(camera_pose_matrix[0:3, 3])

    def gl_display(self):
        self.ui.gl_display(
            self.g_pool.capture.intrinsics.K, self.g_pool.capture.intrinsics.resolution
        )

    def on_resize(self, window, w, h):
        self.ui.on_resize(window, w, h)

    def on_window_key(self, window, key, scancode, action, mods):
        self.ui.on_window_key(window, key, scancode, action, mods)

    def on_close(self, window=None):
        self.ui.on_close(window)

    def on_window_mouse_button(self, window, button, action, mods):
        self.ui.on_window_mouse_button(window, button, action, mods)

    def on_window_pos(self, window, x, y):
        self.ui.on_window_pos(window, x, y)

    def on_scroll(self, window, x, y):
        self.ui.on_scroll(window, x, y)
