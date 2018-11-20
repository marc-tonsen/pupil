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

import marker_tracker_3d.math
import marker_tracker_3d.utils
from marker_tracker_3d.marker_detector import MarkerDetector
from marker_tracker_3d.camera_localizer import CameraLocalizer
from marker_tracker_3d.user_interface import UserInterface
from marker_tracker_3d.camera_model import CameraModel
from marker_tracker_3d import optimization
from marker_tracker_3d.storage import Storage
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

        self.storage = Storage()

        self.marker_detector = MarkerDetector(self.storage)
        self.ui = UserInterface(self, open_3d_window, self.storage)
        self.camera_model = CameraModel()
        self.optimization_controller = optimization.Controller(
            self.storage, self.ui.update_menu
        )

        # for tracking
        self.min_number_of_markers_per_frame_for_loc = 2
        self.marker_model_3d = CameraLocalizer()

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

        self.marker_detector.detect(frame)
        self.optimization_controller.fetch_extrinsics()

        if len(self.storage.markers) < self.min_number_of_markers_per_frame_for_loc:
            self.early_exit()
            return

        if self.marker_model_3d is not None:
            self.update_camera_extrinsics()

        self.optimization_controller.send_marker_data()

    # TODO by now this is doing so little, maybe we should rename it to
    # pop_camera_trace or something similar
    def early_exit(self):
        if len(self.storage.camera_trace):
            self.storage.camera_trace.popleft()

    def update_camera_extrinsics(self):
        self.storage.camera_extrinsics = self.marker_model_3d.current_camera(
            self.storage.markers, self.storage.previous_camera_extrinsics
        )
        if self.storage.camera_extrinsics is None:
            # Do not set previous_camera_extrinsics to None to ensure a decent initial
            # guess for the next solve_pnp call
            self.storage.camera_trace.append(None)
            self.storage.camera_trace_all.append(None)
        else:
            self.storage.previous_camera_extrinsics = self.storage.camera_extrinsics

            camera_pose_matrix = marker_tracker_3d.math.get_camera_pose_mat(
                self.storage.camera_extrinsics
            )
            self.storage.camera_trace.append(camera_pose_matrix[0:3, 3])
            self.storage.camera_trace_all.append(camera_pose_matrix[0:3, 3])

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
