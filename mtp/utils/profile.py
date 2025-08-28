import torch

from contextlib import contextmanager


@contextmanager
def profile_block(name="Code block"):
    torch.cuda.synchronize('cuda')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(torch.cuda.current_stream('cuda'))
    try:
        yield
    finally:
        end.record(torch.cuda.current_stream('cuda'))
        # Synchronize CUDA Kernels before measuring time
        torch.cuda.synchronize('cuda')
        elapsed_time = start.elapsed_time(end) * 1e-3  # CUDA returns ms
        print(f"{name}: {elapsed_time:.4f} s")
