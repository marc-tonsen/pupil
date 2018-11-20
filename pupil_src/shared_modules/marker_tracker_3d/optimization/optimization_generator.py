import threading

from marker_tracker_3d.optimization.visibility_graphs import VisibilityGraphs
from marker_tracker_3d.optimization.optimization import Optimization
from marker_tracker_3d.utils import save_params_dicts


def optimization_generator(recv_pipe):
    first_node_id = None
    graph_for_optimization = VisibilityGraphs(first_node_id=first_node_id)
    event_opt_done = threading.Event()
    event_opt_not_running = threading.Event()
    event_opt_not_running.set()
    lock = threading.RLock()

    while True:
        if recv_pipe.poll(0.001):
            msg, data_recv = recv_pipe.recv()
            if msg == "frame":
                graph_for_optimization.update_visibility_graph_of_keyframes(
                    lock, data_recv
                )

            elif msg == "restart":
                graph_for_optimization = VisibilityGraphs(first_node_id=first_node_id)
                event_opt_done = threading.Event()
                event_opt_not_running = threading.Event()
                event_opt_not_running.set()
                lock = threading.RLock()

            # for experiments
            elif msg == "save":
                dicts = {
                    "marker_extrinsics_opt": graph_for_optimization.marker_extrinsics_opt,
                    "camera_params_opt": graph_for_optimization.camera_params_opt,
                }
                save_path = data_recv
                save_params_dicts(save_path=save_path, dicts=dicts)
                graph_for_optimization.vis_graph(save_path)

        if event_opt_not_running.wait(0.0001):
            event_opt_not_running.clear()
            data_for_optimization = graph_for_optimization.optimization_pre_process(
                lock
            )
            if data_for_optimization is not None:
                opt = Optimization(*data_for_optimization)
                # move Optimization to another thread
                t1 = threading.Thread(
                    name="opt_run", target=opt.run, args=(event_opt_done,)
                )
                t1.start()
            else:
                event_opt_not_running.set()

        if event_opt_done.wait(0.0001):
            event_opt_done.clear()
            result = graph_for_optimization.optimization_post_process(
                lock, opt.result_opt_run
            )
            event_opt_not_running.set()
            if result:
                yield result
