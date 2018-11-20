import collections


class Storage:
    def __init__(self):
        self.markers = {}  # TODO rename to marker_detections
        self.marker_extrinsics = {}

        self.register_new_markers = True

        self.camera_trace = collections.deque(maxlen=100)
        self.camera_trace_all = []

        self.camera_extrinsics = None
        self.previous_camera_extrinsics = None
