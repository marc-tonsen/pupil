import logging
from platform import system

import OpenGL.GL as gl
import cv2
import numpy as np
import pyglui.cygl.utils as pyglui_utils
from pyglui import ui

import gl_utils
import glfw
import square_marker_detect

from marker_tracker_3d import utils
from marker_tracker_3d import math

logger = logging.getLogger(__name__)


class UserInterface:
    def __init__(self, marker_tracker_3d, open_3d_window):
        self.marker_tracker_3d = marker_tracker_3d
        self.open_3d_window = open_3d_window

        self.name = "Marker Tracker 3D"

        self.menu = None

        # window for 3d vis
        if system() == "Linux":
            self.window_position_default = (0, 0)
        elif system() == "Windows":
            self.window_position_default = (8, 31)
        else:
            self.window_position_default = (0, 0)

        self._window = None
        self.trackball = gl_utils.trackball.Trackball()
        self.trackball.zoom_to(-100)
        if self.open_3d_window:
            self.open_window()
        else:
            self.close_window()
        self.scale = 1.0

    def init_ui(self):
        self.marker_tracker_3d.add_menu()
        self.menu = self.marker_tracker_3d.menu
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
                self.marker_tracker_3d.marker_detector,
                step=1,
                min=30,
                max=100,
                label="Perimeter of markers",
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
            ui.Switch(
                "register_new_markers",
                self.marker_tracker_3d,
                label="Registering new markers",
            )
        )
        self.menu.append(
            ui.Button("restart markers registration", self.marker_tracker_3d.restart)
        )
        # TODO external ref
        try:
            self.menu.append(
                ui.Info_Text(
                    "The marker with id {} is defined as the origin of the coordinate system".format(
                        list(
                            self.marker_tracker_3d.marker_model_3d.marker_extrinsics.keys()
                        )[
                            0
                        ]  #
                        # TODO external ref
                    )
                )
            )
        except IndexError:
            self.menu.append(
                ui.Info_Text("The coordinate system has not yet been built up")
            )

        self.menu.append(
            ui.Button("save data", self.marker_tracker_3d.save_data)
        )  # TODO external ref

    def gl_display(self, K, img_size):
        for m in self.marker_tracker_3d.markers.values():
            hat = np.array(
                [[[0, 0], [0, 1], [0.5, 1.3], [1, 1], [1, 0], [0, 0]]], dtype=np.float32
            )
            hat = cv2.perspectiveTransform(
                hat, square_marker_detect.m_marker_to_screen(m)
            )
            pyglui_utils.draw_polyline(
                hat.reshape((6, 2)), color=pyglui_utils.RGBA(0.1, 1.0, 1.0, 0.5)
            )
            pyglui_utils.draw_polyline(
                hat.reshape((6, 2)),
                color=pyglui_utils.RGBA(0.1, 1.0, 1.0, 0.3),
                line_type=gl.GL_POLYGON,
            )

        self.gl_display_in_window_3d(K, img_size)

    def gl_display_in_window_3d(self, K, img_size):
        if self._window:
            active_window = glfw.glfwGetCurrentContext()
            glfw.glfwMakeContextCurrent(self._window)
            gl.glClearColor(0.8, 0.8, 0.8, 1.0)

            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearDepth(1.0)
            gl.glDepthFunc(gl.GL_LESS)
            gl.glEnable(gl.GL_DEPTH_TEST)
            self.trackball.push()

            gl.glMatrixMode(gl.GL_MODELVIEW)
            self.draw_coordinate_system(l=1)

            # Draw registered markers
            for (
                idx,
                extrinsics,
            ) in self.marker_tracker_3d.marker_model_3d.marker_extrinsics.items():
                verts = utils.get_marker_vertex_coord(
                    extrinsics, self.marker_tracker_3d.camera_model
                )
                if idx in self.marker_tracker_3d.markers.keys():
                    color = (1, 0, 0, 0.8)
                else:
                    color = (1, 0.4, 0, 0.6)
                gl.glPushMatrix()
                self.draw_marker(verts, color=color)
                gl.glPopMatrix()

            # Draw camera trace
            if len(self.marker_tracker_3d.camera_trace):
                self.draw_camera_trace(self.marker_tracker_3d.camera_trace)

            # Draw the camera frustum and origin
            if self.marker_tracker_3d.camera_extrinsics is not None:
                camera_pose_matrix = math.get_camera_pose_mat(
                    self.marker_tracker_3d.camera_extrinsics
                )
                gl.glPushMatrix()
                gl.glMultMatrixf(camera_pose_matrix.T.flatten())
                self.draw_frustum(img_size, K, 500)
                gl.glLineWidth(1)
                self.draw_coordinate_system(l=1)
                gl.glPopMatrix()

            self.trackball.pop()

            glfw.glfwSwapBuffers(self._window)
            glfw.glfwMakeContextCurrent(active_window)

    @staticmethod
    def draw_marker(verts, color):
        gl.glColor4f(*color)
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex3f(*verts[0])
        gl.glVertex3f(*verts[1])
        gl.glVertex3f(*verts[1])
        gl.glVertex3f(*verts[2])
        gl.glVertex3f(*verts[2])
        gl.glVertex3f(*verts[3])
        gl.glVertex3f(*verts[3])
        gl.glVertex3f(*verts[0])
        gl.glEnd()

    @staticmethod
    def draw_camera_trace(trace):
        gl.glColor4f(0, 0, 0.8, 0.2)
        for i in range(len(trace) - 1):
            if trace[i] is not None and trace[i + 1] is not None:
                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(*trace[i])
                gl.glVertex3f(*trace[i + 1])
                gl.glEnd()

    @staticmethod
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
        gl.glColor4f(0, 0, 0.6, 0.8)
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(-W, H, Z)
        gl.glVertex3f(W, H, Z)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(W, H, Z)
        gl.glVertex3f(W, -H, Z)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(W, -H, Z)
        gl.glVertex3f(-W, -H, Z)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(-W, -H, Z)
        gl.glVertex3f(-W, H, Z)
        gl.glEnd()

    @staticmethod
    def draw_coordinate_system(l=1):
        # Draw x-axis line.
        gl.glColor3f(1, 0, 0)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(l, 0, 0)
        gl.glEnd()

        # Draw y-axis line.
        gl.glColor3f(0, 1, 0)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(0, l, 0)
        gl.glEnd()

        # Draw z-axis line.
        gl.glColor3f(0, 0, 1)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(0, 0, l)
        gl.glEnd()

    def open_window(self):
        if not self._window:
            monitor = None
            height, width = 1280, 1335

            self._window = glfw.glfwCreateWindow(
                height,
                width,
                self.name,
                monitor=monitor,
                share=glfw.glfwGetCurrentContext(),
            )
            glfw.glfwSetWindowPos(
                self._window,
                self.window_position_default[0],
                self.window_position_default[1],
            )

            self.input = {"down": False, "mouse": (0, 0)}

            # Register callbacks
            glfw.glfwSetFramebufferSizeCallback(self._window, self.on_resize)
            glfw.glfwSetKeyCallback(self._window, self.on_window_key)
            glfw.glfwSetWindowCloseCallback(self._window, self.on_close)
            glfw.glfwSetMouseButtonCallback(self._window, self.on_window_mouse_button)
            glfw.glfwSetCursorPosCallback(self._window, self.on_window_pos)
            glfw.glfwSetScrollCallback(self._window, self.on_scroll)

            self.on_resize(self._window, *glfw.glfwGetFramebufferSize(self._window))

            # gl_state settings
            active_window = glfw.glfwGetCurrentContext()
            glfw.glfwMakeContextCurrent(self._window)
            gl_utils.basic_gl_setup()
            gl_utils.make_coord_system_norm_based()

            # refresh speed settings
            glfw.glfwSwapInterval(0)

            glfw.glfwMakeContextCurrent(active_window)

    def close_window(self):
        if self._window:
            glfw.glfwDestroyWindow(self._window)
            self._window = None

    def on_resize(self, window, w, h):
        self.trackball.set_window_size(w, h)
        active_window = glfw.glfwGetCurrentContext()
        glfw.glfwMakeContextCurrent(window)
        gl_utils.adjust_gl_view(w, h)
        glfw.glfwMakeContextCurrent(active_window)

    def on_window_key(self, window, key, scancode, action, mods):
        if action == glfw.GLFW_PRESS:
            if key == glfw.GLFW_KEY_ESCAPE:
                self.on_close()

    def on_close(self, window=None):
        self.close_window()

    def on_window_mouse_button(self, window, button, action, mods):
        if action == glfw.GLFW_PRESS:
            self.input["down"] = True
            self.input["mouse"] = glfw.glfwGetCursorPos(window)
        if action == glfw.GLFW_RELEASE:
            self.input["down"] = False

    def on_window_pos(self, window, x, y):
        if self.input["down"]:
            old_x, old_y = self.input["mouse"]
            self.trackball.drag_to(x - old_x, y - old_y)
            self.input["mouse"] = x, y

    def on_scroll(self, window, x, y):
        self.trackball.zoom_to(y)

    def deinit_ui(self):
        self.marker_tracker_3d.remove_menu()
