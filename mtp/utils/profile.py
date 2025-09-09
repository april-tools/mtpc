import torch
import time

from contextlib import contextmanager


class TimerResult:
    def __init__(self):
        self.elapsed_time = None


@contextmanager
def time_block(device):

    result = TimerResult()

    if isinstance(device, torch.device):
        device = device.type

    if device == 'cpu':
        start_time = time.perf_counter()
    elif device == 'cuda':
        torch.cuda.synchronize('cuda')
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream('cuda'))
    else:
        raise ValueError('Unknown device  %s' % type(device))
    try:
        yield result
    finally:
        if device == 'cpu':
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
        elif device == 'cuda':
            end.record(torch.cuda.current_stream('cuda'))
            # Synchronize CUDA Kernels before measuring time
            torch.cuda.synchronize('cuda')
            elapsed_time = start.elapsed_time(end) * 1e-3  # CUDA returns ms
        result.elapsed_time = elapsed_time
