import logging

import numpy as np

import square_marker_detect

logger = logging.getLogger(__name__)


class MarkerDetector:
    def __init__(self, storage):
        self.storage = storage
        self.min_marker_perimeter = 100

    def detect(self, frame):
        # not use detect_markers_robust to avoid cv2.calcOpticalFlowPyrLK for
        # performance reasons
        markers = square_marker_detect.detect_markers(
            frame.gray,
            grid_size=5,
            aperture=13,
            min_marker_perimeter=self.min_marker_perimeter,
        )
        markers_dict = self._filter_markers(markers)
        self.storage.markers = markers_dict

    def _filter_markers(self, markers):
        markers_id_all = set([m["id"] for m in markers])
        for marker_id in markers_id_all:
            markers_with_same_id = [m for m in markers if m["id"] == marker_id]
            if len(markers_with_same_id) > 2:
                markers = [m for m in markers if m["id"] != marker_id]
                logger.warning(
                    "WARNING! Multiple markers with same id {} found!".format(marker_id)
                )
            elif len(markers_with_same_id) == 2:
                markers = self._remove_duplicate(
                    marker_id, markers, markers_with_same_id
                )

        marker_dict = {
            m["id"]: {k: v for k, v in m.items() if k != "id"} for m in markers
        }

        return marker_dict

    def _remove_duplicate(self, marker_id, markers, markers_with_same_id):
        dist = np.linalg.norm(
            np.array(markers_with_same_id[0]["centroid"])
            - np.array(markers_with_same_id[1]["centroid"])
        )
        # If two markers are very close, pick the bigger one. It may due to double detection
        if dist < 3:
            marker_small = min(markers_with_same_id, key=lambda x: x["perimeter"])
            markers = [
                m
                for m in markers
                if not (
                    m["id"] == marker_id and m["centroid"] == marker_small["centroid"]
                )
            ]
        else:
            markers = [m for m in markers if m["id"] != marker_id]
            logger.warning(
                "WARNING! Multiple markers with same id {} found!".format(marker_id)
            )
        return markers
