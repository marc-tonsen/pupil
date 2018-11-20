import numpy as np

from marker_tracker_3d.utils import merge_param, check_camera_param
from marker_tracker_3d.camera_model import CameraModel


class CameraLocalizer(CameraModel):
    def __init__(self, marker_extrinsics=None):
        super().__init__()
        if marker_extrinsics is None:
            self.marker_extrinsics = {}
        else:
            self.marker_extrinsics = marker_extrinsics

    @property
    def marker_extrinsics(self):
        return self.__marker_extrinsics

    @marker_extrinsics.setter
    def marker_extrinsics(self, marker_extrinsics_new):
        assert isinstance(marker_extrinsics_new, dict)
        self.__marker_extrinsics = marker_extrinsics_new

    def prepare_data_for_localization(self, current_frame):
        marker_keys_available = current_frame.keys() & set(
            self.marker_extrinsics.keys()
        )

        marker_points_3d_for_loc = self.params_to_points_3d(
            [self.marker_extrinsics[i] for i in marker_keys_available]
        )
        marker_points_2d_for_loc = np.array(
            [current_frame[i]["verts"] for i in marker_keys_available]
        )

        if len(marker_points_3d_for_loc) and len(marker_points_2d_for_loc):
            marker_points_3d_for_loc.shape = 1, -1, 3
            marker_points_2d_for_loc.shape = 1, -1, 2

        return marker_points_3d_for_loc, marker_points_2d_for_loc

    def current_camera(self, current_frame, camera_params_loc_prv=None):
        marker_points_3d_for_loc, marker_points_2d_for_loc = self.prepare_data_for_localization(
            current_frame
        )

        retval, rvec, tvec = self.run_solvePnP(
            marker_points_3d_for_loc, marker_points_2d_for_loc, camera_params_loc_prv
        )

        if retval:
            if check_camera_param(marker_points_3d_for_loc, rvec, tvec):
                camera_params_loc = merge_param(rvec, tvec)
                return camera_params_loc
