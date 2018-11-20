import multiprocessing as mp
import background_helper
import logging

from marker_tracker_3d import optimization

logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, storage, on_first_yield=None):
        self.storage = storage
        self.on_first_yield = on_first_yield
        self.first_yield_done = False
        self.frame_count = 0
        self.send_data_interval = 6

        recv_pipe, self.send_pipe = mp.Pipe(False)
        generator_args = (recv_pipe,)
        self.bg_task = background_helper.IPC_Logging_Task_Proxy(
            name="generator_optimization",
            generator=optimization.optimization_generator,
            args=generator_args,
        )

    def fetch_extrinsics(self):
        for marker_extrinsics in self.bg_task.fetch():
            if not self.first_yield_done:
                self.on_first_yield()
                self.first_yield_done = True

            self.storage.marker_extrinsics = marker_extrinsics

            logger.info(
                "{} markers have been registered and updated".format(
                    len(marker_extrinsics)
                )
            )

    def send_marker_data(self):
        self.frame_count += 1
        if self.frame_count > self.send_data_interval:
            self.frame_count = 0
            if self.storage.register_new_markers:
                # TODO I need to use camera_extrinsics or marker_extrinsics here?
                self.send_pipe.send(
                    ("frame", (self.storage.markers, self.storage.camera_extrinsics))
                )
