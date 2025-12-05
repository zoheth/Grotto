"""
Profiling utilities for tracking GPU performance.
"""

from contextlib import contextmanager
from typing import Optional

import torch

# Direct export of torch.profiler.record_function for module tracking
record_module = torch.profiler.record_function


@contextmanager
def profiling_session(output_path: Optional[str] = None):
    """
    Context manager for a profiling session with torch profiler.

    Args:
        output_path: Optional path to save the Chrome trace file (without extension).

    Example:
        with profiling_session("profile_trace/grotto_251201_1430"):
            model(input)
    """
    # Need both CPU and CUDA for record_function labels to show
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
    ) as prof:
        yield prof

    if output_path:
        prof.export_chrome_trace(f"{output_path}.json")
