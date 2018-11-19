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
from platform import system

import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import GL_LINES
from pyglui import ui
from pyglui.cygl.utils import draw_polyline, RGBA

import background_helper as bh
import marker_tracker_3d.generator_optimization
import marker_tracker_3d.math
import marker_tracker_3d.utils
import square_marker_detect
from gl_utils import adjust_gl_view, basic_gl_setup, make_coord_system_norm_based
from gl_utils.trackball import Trackball
from glfw import *
from marker_tracker_3d import utils
from marker_tracker_3d.marker_model_3d import MarkerModel3D
from plugin import Plugin

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)


class Marker_Tracker_3D(Plugin):
    """
    This plugin tracks the pose of the scene camera based on fiducial markers in the environment.
    """

    icon_chr = chr(0xEC07)
    icon_font = "pupil_icons"

    def __init__(
        self,
        g_pool,
        open_3d_window=True,
        min_id_confidence=0.9,
        min_marker_perimeter=100,
    ):
        super().__init__(g_pool)
        # for UI menu
        self.open_3d_window = open_3d_window
        self.min_id_confidence = min_id_confidence
        self.min_marker_perimeter = min_marker_perimeter

        self.name = "Marker Tracker 3D"

        # window for 3d vis
        if system() == "Linux":
            self.window_position_default = (0, 0)
        elif system() == "Windows":
            self.window_position_default = (8, 31)
        else:
            self.window_position_default = (0, 0)
        self._window = None
        self.fullscreen = False
        self.trackball = Trackball()
        self.trackball.zoom_to(-100)
        if self.open_3d_window:
            self.open_window()
        else:
            self.close_window()
        self.scale = 1.0

        # for tracking
        self.send_data_interval = 6
        self.min_number_of_markers_per_frame_for_loc = 2
        self.registering = True
        self.markers_drawn_in_3d_window = list()
        self.markers = list()
        self.marker_model_3D = None
        self.camera_pose = None
        self.camera_trace = collections.deque(maxlen=100)
        self.camera_params_loc = None
        self.first_node = None
        self.frame_count = 0
        self.frame_count_last_send_data = 0
        self.square_params_opt = dict()

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
        self.add_menu()
        self.menu.label = "3D Marker Tracker"
        self.update_menu()

    def update_menu(self):
        def open_close_window(open_3d_window):
            self.open_3d_window = open_3d_window
            if self.open_3d_window:
                self.open_window()
                logger.info("3d visualization window is opened")
            else:
                self.close_window()
                logger.info("3d visualization window is closed")

        self.menu.elements[:] = list()
        self.menu.append(ui.Info_Text("This plugin detects current camera pose"))
        self.menu.append(
            ui.Slider(
                "min_marker_perimeter",
                self,
                step=1,
                min=30,
                max=100,
                label="Perimeter of markers",
            )
        )
        self.menu.append(
            ui.Slider(
                "min_id_confidence",
                self,
                step=0.05,
                min=0,
                max=1,
                label="Confidence of marker detection",
            )
        )
        self.menu.append(
            ui.Switch(
                "open_3d_window",
                self,
                setter=open_close_window,
                label="3d visualization window",
            )
        )
        self.menu.append(
            ui.Switch("registering", self, label="Registering new markers")
        )
        self.menu.append(ui.Button("restart markers registration", self.restart))
        if self.first_node is not None:
            self.menu.append(
                ui.Info_Text(
                    "The marker with id {} is defined as the origin of the coordinate system".format(
                        self.first_node
                    )
                )
            )
        else:
            self.menu.append(
                ui.Info_Text("The coordinate system has not yet been built up")
            )

        self.menu.append(ui.Button("save data", self.save_data))

    def deinit_ui(self):
        self.remove_menu()

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
        self.update_menu()
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
        self.registering = True
        self.markers_drawn_in_3d_window = list()
        self.markers = list()
        self.marker_model_3D = None
        self.camera_pose = None
        self.camera_trace = collections.deque(maxlen=100)
        self.camera_params_loc = None
        self.first_node = None
        self.frame_count = 0
        self.frame_count_last_send_data = 0
        self.square_params_opt = dict()

        # for experiments
        self.robustness = list()
        self.camera_trace_all = list()
        self.all_frames = list()
        self.reprojection_errors = list()

    def get_init_dict(self):
        d = super().get_init_dict()
        d["open_3d_window"] = self.open_3d_window
        d["min_id_confidence"] = self.min_id_confidence
        d["min_marker_perimeter"] = self.min_marker_perimeter
        return d

    def recent_events(self, events):
        # Get current frame
        frame = events.get("frame")
        if not frame:
            return

        self.fetch_data_from_bg_optimization()

        self.markers = self.detect_and_filter_markers(frame)

        if len(self.markers) < self.min_number_of_markers_per_frame_for_loc:
            self.camera_pose = None
            if len(self.camera_trace):
                self.camera_trace.popleft()
            return

        if self.marker_model_3D is None:
            if (
                self.frame_count - self.frame_count_last_send_data
                >= self.send_data_interval
            ):
                self.frame_count_last_send_data = self.frame_count
                # if not registering, not send data
                if self.registering:
                    self.send_pipe.send(("frame", (self.markers, None)))
        else:
            camera_params_loc = self.marker_model_3D.current_camera(
                self.markers, self.camera_params_loc
            )

            if isinstance(camera_params_loc, np.ndarray):
                self.camera_params_loc = camera_params_loc
                self.camera_pose = marker_tracker_3d.math.get_camera_pose_mat(
                    self.camera_params_loc
                )
                self.camera_trace.append(self.camera_pose[0:3, 3])
                self.camera_trace_all.append(self.camera_pose[0:3, 3])

                # send current_frame through pipe
                if self.frame_count - self.frame_count_last_send_data >= 5:
                    self.frame_count_last_send_data = self.frame_count
                    # if not registering, not send data
                    if self.registering:
                        self.send_pipe.send(
                            ("frame", (self.markers, camera_params_loc))
                        )
            else:
                self.camera_pose = None
                self.camera_trace.append(None)
                self.camera_trace_all.append(None)

        self.frame_count += 1

    def fetch_data_from_bg_optimization(self):
        for data in self.bg_task.fetch():
            assert isinstance(data, tuple) and len(data) == 3
            # TODO delete assertion
            self.square_params_opt, self.markers_drawn_in_3d_window, first_node = data
            if self.first_node is None:
                self.first_node = first_node
                self.update_menu()

            try:
                self.marker_model_3D.square_params_opt = self.square_params_opt
            except AttributeError:
                self.marker_model_3D = MarkerModel3D(self.square_params_opt)

            logger.info(
                "{} markers have been registered and updated".format(
                    len(self.square_params_opt)
                )
            )

    def detect_and_filter_markers(self, frame):
        # not use detect_markers_robust to avoid cv2.calcOpticalFlowPyrLK for
        # performance reasons
        markers = square_marker_detect.detect_markers(
            frame.gray,
            grid_size=5,
            aperture=13,
            min_marker_perimeter=self.min_marker_perimeter,
        )
        markers_dict = utils.filter_markers(markers)
        return markers_dict

    def gl_display(self):
        for m in self.markers.values():
            hat = np.array(
                [[[0, 0], [0, 1], [0.5, 1.3], [1, 1], [1, 0], [0, 0]]], dtype=np.float32
            )
            hat = cv2.perspectiveTransform(
                hat, square_marker_detect.m_marker_to_screen(m)
            )
            if (
                m["perimeter"] >= self.min_marker_perimeter
                and m["id_confidence"] > self.min_id_confidence
            ):
                draw_polyline(hat.reshape((6, 2)), color=RGBA(0.1, 1.0, 1.0, 0.5))
                draw_polyline(
                    hat.reshape((6, 2)),
                    color=RGBA(0.1, 1.0, 1.0, 0.3),
                    line_type=GL_POLYGON,
                )
            else:
                draw_polyline(hat.reshape((6, 2)), color=RGBA(0.1, 1.0, 1.0, 0.5))

        # 3D debug visualization
        self.gl_display_in_window_3d(self.g_pool.image_tex)

    def gl_display_in_window_3d(self, world_tex):
        """
        Display camera pose and markers in 3D space
        """

        K, img_size = (
            self.g_pool.capture.intrinsics.K,
            self.g_pool.capture.intrinsics.resolution,
        )

        if self._window:
            active_window = glfwGetCurrentContext()
            glfwMakeContextCurrent(self._window)
            glClearColor(0.8, 0.8, 0.8, 1.0)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearDepth(1.0)
            glDepthFunc(GL_LESS)
            glEnable(GL_DEPTH_TEST)
            self.trackball.push()

            glMatrixMode(GL_MODELVIEW)
            draw_coordinate_system(l=1)

            # Draw registered markers
            for verts, idx in zip(
                self.markers_drawn_in_3d_window, self.square_params_opt.keys()
            ):
                if idx in self.markers.keys():
                    color = (1, 0, 0, 0.8)
                else:
                    color = (1, 0.4, 0, 0.6)
                glPushMatrix()
                draw_marker(verts, color=color)
                glPopMatrix()

            # Draw camera trace
            if len(self.camera_trace):
                draw_camera_trace(self.camera_trace)

            # Draw the camera frustum and origin
            if self.camera_pose is not None:
                glPushMatrix()
                glMultMatrixf(self.camera_pose.T.flatten())
                draw_frustum(img_size, K, 500)
                glLineWidth(1)
                draw_coordinate_system(l=1)
                glPopMatrix()

            self.trackball.pop()

            glfwSwapBuffers(self._window)
            glfwMakeContextCurrent(active_window)

    def open_window(self):
        if not self._window:
            if self.fullscreen:
                monitor = glfwGetMonitors()[self.monitor_idx]
                mode = glfwGetVideoMode(monitor)
                height, width = mode[0], mode[1]
            else:
                monitor = None
                height, width = 1280, 1335

            self._window = glfwCreateWindow(
                height, width, self.name, monitor=monitor, share=glfwGetCurrentContext()
            )
            if not self.fullscreen:
                glfwSetWindowPos(
                    self._window,
                    self.window_position_default[0],
                    self.window_position_default[1],
                )

            self.input = {"down": False, "mouse": (0, 0)}

            # Register callbacks
            glfwSetFramebufferSizeCallback(self._window, self.on_resize)
            glfwSetKeyCallback(self._window, self.on_window_key)
            glfwSetWindowCloseCallback(self._window, self.on_close)
            glfwSetMouseButtonCallback(self._window, self.on_window_mouse_button)
            glfwSetCursorPosCallback(self._window, self.on_window_pos)
            glfwSetScrollCallback(self._window, self.on_scroll)

            self.on_resize(self._window, *glfwGetFramebufferSize(self._window))

            # gl_state settings
            active_window = glfwGetCurrentContext()
            glfwMakeContextCurrent(self._window)
            basic_gl_setup()
            make_coord_system_norm_based()

            # refresh speed settings
            glfwSwapInterval(0)

            glfwMakeContextCurrent(active_window)

    def close_window(self):
        if self._window:
            glfwDestroyWindow(self._window)
            self._window = None

    # window calbacks
    def on_resize(self, window, w, h):
        self.trackball.set_window_size(w, h)
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(window)
        adjust_gl_view(w, h)
        glfwMakeContextCurrent(active_window)

    def on_window_key(self, window, key, scancode, action, mods):
        if action == GLFW_PRESS:
            if key == GLFW_KEY_ESCAPE:
                self.on_close()

    def on_close(self, window=None):
        self.close_window()

    def on_window_mouse_button(self, window, button, action, mods):
        if action == GLFW_PRESS:
            self.input["down"] = True
            self.input["mouse"] = glfwGetCursorPos(window)
        if action == GLFW_RELEASE:
            self.input["down"] = False

    def on_window_pos(self, window, x, y):
        if self.input["down"]:
            old_x, old_y = self.input["mouse"]
            self.trackball.drag_to(x - old_x, y - old_y)
            self.input["mouse"] = x, y

    def on_scroll(self, window, x, y):
        self.trackball.zoom_to(y)


def draw_marker(verts, color):
    glColor4f(*color)
    glBegin(GL_LINE_LOOP)
    glVertex3f(*verts[0])
    glVertex3f(*verts[1])
    glVertex3f(*verts[1])
    glVertex3f(*verts[2])
    glVertex3f(*verts[2])
    glVertex3f(*verts[3])
    glVertex3f(*verts[3])
    glVertex3f(*verts[0])
    glEnd()


def draw_camera_trace(trace):
    glColor4f(0, 0, 0.8, 0.2)
    for i in range(len(trace) - 1):
        if trace[i] is not None and trace[i + 1] is not None:
            glBegin(GL_LINES)
            glVertex3f(*trace[i])
            glVertex3f(*trace[i + 1])
            glEnd()


def draw_frustum(img_size, K, scale=1):
    # average focal length
    f = (K[0, 0] + K[1, 1]) / 2
    # compute distances for setting up the camera pyramid
    W = 0.5 * (img_size[0])
    H = 0.5 * (img_size[1])
    Z = f
    # scale the pyramid
    W /= scale
    H /= scale
    Z /= scale
    # draw it
    glColor4f(0, 0, 0.6, 0.8)
    glBegin(GL_LINE_LOOP)
    glVertex3f(0, 0, 0)
    glVertex3f(-W, H, Z)
    glVertex3f(W, H, Z)
    glVertex3f(0, 0, 0)
    glVertex3f(W, H, Z)
    glVertex3f(W, -H, Z)
    glVertex3f(0, 0, 0)
    glVertex3f(W, -H, Z)
    glVertex3f(-W, -H, Z)
    glVertex3f(0, 0, 0)
    glVertex3f(-W, -H, Z)
    glVertex3f(-W, H, Z)
    glEnd()


def draw_coordinate_system(l=1):
    # Draw x-axis line.
    glColor3f(1, 0, 0)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(l, 0, 0)
    glEnd()

    # Draw y-axis line.
    glColor3f(0, 1, 0)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(0, l, 0)
    glEnd()

    # Draw z-axis line.
    glColor3f(0, 0, 1)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, l)
    glEnd()
